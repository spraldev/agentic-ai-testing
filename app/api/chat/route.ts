import { NextRequest } from "next/server"

import { GoogleGenerativeAI } from "@google/generative-ai"
import OpenAI from "openai"
import Anthropic from "@anthropic-ai/sdk"

// ============================================================================
// Type Definitions
// ============================================================================

type ModelId = 'claude' | 'gpt' | 'gemini'

interface ModelConfig {
  id: ModelId
  provider: 'openai' | 'anthropic' | 'google'
  modelName: string
}

interface Round1Answer {
  modelId: ModelId
  rawAnswer: string
  reasoning: string
  finalAnswer: string
}

interface Round2Critique {
  modelId: ModelId
  rawCritique: string
  critique: string
  revisedAnswer: string
}

interface DebateResult {
  question: string
  round1: Round1Answer[]
  round2: Round2Critique[]
  finalAnswer: string
  finalRationale: string
  chosenFrom: ModelId[]
}

// ============================================================================
// Model Configurations
// ============================================================================

const MODELS: ModelConfig[] = [
  { id: 'claude', provider: 'anthropic', modelName: 'claude-3-haiku-20240307' },
  { id: 'gpt', provider: 'openai', modelName: 'gpt-3.5-turbo' },
  { id: 'gemini', provider: 'google', modelName: 'gemini-2.5-flash' }
]

const ARBITER_MODEL: ModelConfig = MODELS[2] // Use Gemini as arbiter

// ============================================================================
// Prompt Templates
// ============================================================================

const R1_SYSTEM_PROMPT = `You are an AI assistant solving a problem independently.

1. Think step-by-step to reach an answer (be CONCISE - 2-3 sentences max).
2. Write your reasoning in a block labeled REASONING:
3. Then write a single final answer on one line labeled FINAL_ANSWER:

Use this output format exactly:

REASONING: <your concise reasoning (2-3 sentences)>

FINAL_ANSWER: <your one-line final answer here>`

const R2_SYSTEM_PROMPT = `You are participating in a multi-model debate.

You will see:
- The original QUESTION
- YOUR_PREVIOUS_ANSWER
- ANSWERS_FROM_OTHERS

Your job:
1. Briefly critique where answers differ (1-2 sentences max).
2. Decide the best final answer.
3. Update your answer if needed.

Use this exact output format:

CRITIQUE:
<brief critique (1-2 sentences), refer to models by id>

UPDATED_FINAL_ANSWER: <your one-line final answer after considering the debate>

Be CONCISE. Do NOT invent new questions.`

const R3_SYSTEM_PROMPT = `You are an arbiter reading the results of a multi-model debate.

You will see the QUESTION and candidate answers from 3 models across 2 rounds.

Your job:
1. Review all UPDATED_FINAL_ANSWER values from Round 2.
2. Choose the most correct answer.
3. Synthesize if models agree.

Output format:

FINAL_ANSWER: <your one-line final answer>

RATIONALE:
<1-2 sentences explaining your choice, reference model ids>

Be CONCISE. Choose the best answer even if uncertain.`

// ============================================================================
// Text Parsing Utilities
// ============================================================================

function extractBetween(text: string, startMarker: string, endMarker: string): string {
  const startIdx = text.indexOf(startMarker)
  if (startIdx === -1) return ''

  const contentStart = startIdx + startMarker.length
  const endIdx = text.indexOf(endMarker, contentStart)

  if (endIdx === -1) {
    return text.substring(contentStart).trim()
  }

  return text.substring(contentStart, endIdx).trim()
}

function extractAfter(text: string, marker: string): string {
  const idx = text.indexOf(marker)
  if (idx === -1) return ''

  const contentStart = idx + marker.length
  const restOfText = text.substring(contentStart).trim()

  // Extract only the first line for single-line answers
  const firstLineEnd = restOfText.indexOf('\n')
  if (firstLineEnd === -1) return restOfText

  return restOfText.substring(0, firstLineEnd).trim()
}

// ============================================================================
// LLM Client Abstraction
// ============================================================================

async function callLLM(
  config: ModelConfig,
  systemPrompt: string,
  userPrompt: string,
  clients: {
    anthropic: Anthropic
    openai: OpenAI
    genAI: GoogleGenerativeAI
  },
  maxTokens: number = 512
): Promise<string> {
  try {
    switch (config.provider) {
      case 'anthropic': {
        const response = await clients.anthropic.messages.create({
          model: config.modelName,
          max_tokens: maxTokens,
          messages: [
            {
              role: 'user',
              content: `${systemPrompt}\n\n${userPrompt}`
            }
          ]
        })
        return response.content[0].type === 'text' ? response.content[0].text : ''
      }

      case 'openai': {
        const response = await clients.openai.chat.completions.create({
          model: config.modelName,
          max_tokens: maxTokens,
          messages: [
            { role: 'system', content: systemPrompt },
            { role: 'user', content: userPrompt }
          ]
        })
        return response.choices[0]?.message?.content || ''
      }

      case 'google': {
        const model = clients.genAI.getGenerativeModel({
          model: config.modelName,
          systemInstruction: systemPrompt,
          generationConfig: {
            maxOutputTokens: maxTokens
          }
        })
        const result = await model.generateContent(userPrompt)
        return result.response.text()
      }

      default:
        throw new Error(`Unknown provider: ${config.provider}`)
    }
  } catch (error) {
    console.error(`Error calling ${config.id}:`, error)
    throw error
  }
}

// ============================================================================
// Round 1: Independent Solutions (Parallel)
// ============================================================================

async function runRound1(
  question: string,
  clients: {
    anthropic: Anthropic
    openai: OpenAI
    genAI: GoogleGenerativeAI
  }
): Promise<Round1Answer[]> {
  const userPrompt = `QUESTION:\n${question}`

  const results = await Promise.all(
    MODELS.map(async (model) => {
      const responseText = await callLLM(model, R1_SYSTEM_PROMPT, userPrompt, clients)

      const reasoning = extractBetween(responseText, 'REASONING:', 'FINAL_ANSWER:')
      const finalAnswer = extractAfter(responseText, 'FINAL_ANSWER:')

      console.log(`Round 1 - ${model.id}:`, {
        reasoning: reasoning.substring(0, 100) + '...',
        finalAnswer
      })

      return {
        modelId: model.id,
        rawAnswer: responseText,
        reasoning: reasoning || responseText, // fallback to full text if marker missing
        finalAnswer: finalAnswer || 'No answer provided'
      }
    })
  )

  return results
}

// ============================================================================
// Round 2: Critique & Revise (Parallel)
// ============================================================================

function buildRound2UserPrompt(
  question: string,
  self: Round1Answer,
  others: Round1Answer[]
): string {
  // Truncate reasoning to 300 chars max to speed up processing
  const truncate = (text: string) => text.substring(0, 300) + (text.length > 300 ? '...' : '')

  const othersText = others
    .map(
      (o) => `ANSWER_FROM_${o.modelId}:
REASONING:
${truncate(o.reasoning)}

FINAL_ANSWER:
${o.finalAnswer}`
    )
    .join('\n\n')

  return `QUESTION:
${question}

YOUR_PREVIOUS_ANSWER (${self.modelId}):
REASONING:
${truncate(self.reasoning)}

FINAL_ANSWER:
${self.finalAnswer}

${othersText}`
}

async function runRound2(
  question: string,
  round1: Round1Answer[],
  clients: {
    anthropic: Anthropic
    openai: OpenAI
    genAI: GoogleGenerativeAI
  }
): Promise<Round2Critique[]> {
  const results = await Promise.all(
    round1.map(async (self) => {
      const others = round1.filter((r) => r.modelId !== self.modelId)
      const userPrompt = buildRound2UserPrompt(question, self, others)

      const modelConfig = MODELS.find((m) => m.id === self.modelId)!
      const responseText = await callLLM(modelConfig, R2_SYSTEM_PROMPT, userPrompt, clients)

      const critique = extractBetween(responseText, 'CRITIQUE:', 'UPDATED_FINAL_ANSWER:')
      const revisedAnswer = extractAfter(responseText, 'UPDATED_FINAL_ANSWER:')

      console.log(`Round 2 - ${self.modelId}:`, {
        critique: critique.substring(0, 100) + '...',
        revisedAnswer
      })

      return {
        modelId: self.modelId,
        rawCritique: responseText,
        critique: critique || responseText,
        revisedAnswer: revisedAnswer || self.finalAnswer // fallback to R1 answer
      }
    })
  )

  return results
}

// ============================================================================
// Round 3: Arbiter Synthesis
// ============================================================================

function buildArbiterPrompt(
  question: string,
  round1: Round1Answer[],
  round2: Round2Critique[]
): string {
  // Truncate text to 200 chars max to speed up arbiter processing
  const truncate = (text: string) => text.substring(0, 200) + (text.length > 200 ? '...' : '')

  let text = `QUESTION:
${question}

ROUND_1_ANSWERS:
`

  for (const r1 of round1) {
    text += `
MODEL: ${r1.modelId}
FINAL_ANSWER:
${r1.finalAnswer}
REASONING:
${truncate(r1.reasoning)}
`
  }

  text += `

ROUND_2_UPDATED_ANSWERS:
`

  for (const r2 of round2) {
    text += `
MODEL: ${r2.modelId}
UPDATED_FINAL_ANSWER:
${r2.revisedAnswer}
CRITIQUE:
${truncate(r2.critique)}
`
  }

  return text
}

async function runRound3(
  question: string,
  round1: Round1Answer[],
  round2: Round2Critique[],
  clients: {
    anthropic: Anthropic
    openai: OpenAI
    genAI: GoogleGenerativeAI
  }
): Promise<{ finalAnswer: string; rationale: string }> {
  const arbiterPrompt = buildArbiterPrompt(question, round1, round2)

  const responseText = await callLLM(ARBITER_MODEL, R3_SYSTEM_PROMPT, arbiterPrompt, clients, 768)

  const finalAnswer = extractAfter(responseText, 'FINAL_ANSWER:')
  const rationale = extractAfter(responseText, 'RATIONALE:')

  console.log('Round 3 - Arbiter:', {
    finalAnswer,
    rationale: rationale.substring(0, 100) + '...'
  })

  return {
    finalAnswer: finalAnswer || 'Unable to determine final answer',
    rationale: rationale || responseText
  }
}

// ============================================================================
// Main API Handler
// ============================================================================

export async function POST(req: NextRequest) {
  try {
    const { question, debug } = (await req.json()) as { question: string; debug?: boolean }

    if (!question || typeof question !== 'string') {
      return new Response(JSON.stringify({ error: 'Missing or invalid question' }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' }
      })
    }

    // Initialize AI clients
    const clients = {
      anthropic: new Anthropic({
        apiKey: process.env.ANTHROPIC_API_KEY
      }),
      openai: new OpenAI({
        apiKey: process.env.OPENAI_API_KEY
      }),
      genAI: new GoogleGenerativeAI(process.env.GOOGLE_API_KEY || '')
    }

    console.log('\n=== Starting Multi-Model Debate ===')
    console.log('Question:', question)

    // Execute the three rounds
    const round1 = await runRound1(question, clients)
    const round2 = await runRound2(question, round1, clients)
    const { finalAnswer, rationale } = await runRound3(question, round1, round2, clients)

    const result: DebateResult = {
      question,
      round1,
      round2,
      finalAnswer,
      finalRationale: rationale,
      chosenFrom: round2.map((r) => r.modelId)
    }

    console.log('=== Debate Complete ===\n')

    return new Response(JSON.stringify(result, null, debug ? 2 : 0), {
      status: 200,
      headers: {
        'Content-Type': 'application/json'
      }
    })
  } catch (error) {
    console.error('Chat API error:', error)
    return new Response(
      JSON.stringify({
        error: 'Internal server error',
        message: error instanceof Error ? error.message : 'Unknown error'
      }),
      {
        status: 500,
        headers: { 'Content-Type': 'application/json' }
      }
    )
  }
}
