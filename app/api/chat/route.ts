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

const ARBITER_MODEL: ModelConfig = MODELS[0] // Use Claude as arbiter (more reliable)

// ============================================================================
// Prompt Templates
// ============================================================================

const R1_SYSTEM_PROMPT = `Solve this task. Quality and correctness matter more than speed.

STEP 1 - UNDERSTAND BEFORE STARTING:
• Restate what's being asked in your own words
• If examples/test cases exist: work through Example 1 BY HAND first and explain why it produces that output
• If no examples exist: create a simple test case yourself to verify your understanding
• What makes this tricky? What could go wrong?

STEP 2 - PLAN, THEN EXECUTE:
Outline your approach briefly:
• For code: algorithm, data structures, time complexity
• For math: technique, formulas, key steps
• For writing: thesis/main argument and 3-4 supporting points
• For analysis: framework, criteria, structure

Then execute fully with clear reasoning shown.

STEP 3 - VERIFY YOUR WORK:
You MUST check your answer before finalizing.

For CODE:
→ Trace through Example 1 step-by-step, showing variable values at each line
→ Does your output match expected? If NO: stop, find the bug, fix it, re-verify
→ CRITICAL: If you sorted any array, did you keep related data paired with it?

For MATH:
→ Redo key calculations to double-check arithmetic
→ Do units/dimensions make sense? Is the magnitude reasonable?
→ If data was reordered, did you maintain necessary pairings?

For WRITING/ANALYSIS:
→ Read the first sentence of each paragraph in sequence — do they tell a logical story?
→ Does every paragraph support your main thesis/argument?
→ Is every claim backed by evidence, example, or reasoning?

For ALL TASKS:
→ Did you actually answer what was asked, or did you drift?

STEP 4 - STRESS TEST:
Assume your answer has a flaw. Try to find it:
• What's the weakest part?
• What would a critic attack?
• For code/math: test an edge case (n=0, n=1, negative values, huge values)
• For writing: what would someone who disagrees say? Is it addressed?

If you find a problem → fix it before finalizing.

OUTPUT FORMAT:
UNDERSTANDING: [Task restated + Example 1 traced by hand (or your own test case) + what makes it tricky]
APPROACH: [Your plan]
SOLUTION: [Complete answer with reasoning shown]
VERIFICATION: [What you checked + trace/evidence + results]
FINAL_ANSWER: [Your final answer]
CONFIDENCE: [High / Medium / Low — with brief explanation of uncertainties]`

const R2_SYSTEM_PROMPT = `You are reviewing solutions for correctness. You will see:
• The original task
• Your Round 1 answer
• Other models' Round 1 answers (if available — some may have failed)

YOUR GOAL: Find the correct answer. Do not assume any solution is right — including your own.

⚠️ CRITICAL WARNING: Agreement does NOT mean correctness. If all solutions agree, be EXTRA skeptical — they might all share the same blind spot. This is common and dangerous.

STEP 1 - HANDLE MISSING/FAILED SOLUTIONS:
If any model failed to produce an answer (API error, refusal, "no answer provided"), note it and ignore that model. Work with what you have.

STEP 2 - TEST EACH SOLUTION:
For each available solution, verify it independently. Show your work.

For CODE:
→ Trace through Example 1 line by line, showing variable values
→ Does output match expected? Mark: ✓ PASS or ✗ FAIL
→ CRITICAL: If any array was sorted, did related data stay paired with it?

For MATH:
→ Redo the key calculations yourself
→ Does your result match theirs? Mark: ✓ PASS or ✗ FAIL
→ Check: If data was reordered, are pairings preserved?

For WRITING/ANALYSIS:
→ State the thesis/main argument in one sentence
→ Does every paragraph support it? Is each claim evidenced?
→ Mark: ✓ SOUND or ✗ FLAWED (state why)

For ALL:
→ Does it actually answer what was asked?

STEP 3 - TRY TO BREAK EACH SOLUTION:
Even if a solution passed Step 2, actively try to find a flaw:
• What edge case might fail? (n=0, n=1, negatives, huge values)
• What's the weakest assumption or argument?
• Can you construct a specific input or counterargument that breaks it?

STEP 4 - CHECK FOR SHARED BLIND SPOTS:
If solutions AGREE:
→ This is suspicious. Ask: "What could we ALL be getting wrong?"
→ Re-examine the shared core logic independently
→ Verify shared assumptions explicitly — don't just accept them

If solutions DISAGREE:
→ Identify the exact point of disagreement
→ Test that specific point to determine who's right

STEP 5 - REVIEW CONFIDENCE SIGNALS:
Look at the confidence levels from Round 1:
• If one model said "High confidence" and another said "Low confidence," weigh the high-confidence answer more — but still verify it
• If a model expressed specific uncertainties, check if those concerns are valid

STEP 6 - DELIVER VERDICT:
Choose one:
a) One solution is clearly correct → select it, explain why
b) Multiple are correct → pick the best reasoned/cleanest
c) All are flawed → identify the error and provide a corrected answer with work shown
d) Genuinely uncertain → state what's unresolved

OUTPUT FORMAT:
SOLUTIONS_REVIEWED: [List which models you're evaluating, note any that failed/were ignored]
VERIFICATION: [For each solution: your trace/analysis + ✓/✗ verdict with reasoning]
BREAK_ATTEMPTS: [Edge cases, counterarguments, and adversarial inputs you tested]
BLIND_SPOT_CHECK: [If solutions agreed: what might they all be missing? What did you verify? If they disagreed: who's right on the disputed point?]
CRITIQUE: [Summary: what's right, what's wrong, who you trust and why]
UPDATED_FINAL_ANSWER: [The correct answer — either selected or corrected]
CONFIDENCE: [High / Medium / Low — and what remains uncertain]`

const R3_SYSTEM_PROMPT = `You are the final judge. Your ruling is final. You will see:
• The original task
• All Round 1 solutions
• All Round 2 critiques and revised answers

YOUR MISSION: Deliver the correct answer. No one reviews your work — get it right.

⚠️ JUDGING PRINCIPLES:

1. VERIFICATION BEATS VOTING
   If 3 models agree but your verification shows they're wrong, they're wrong.
   Do not be swayed by consensus. Trust your own verification.

2. AGREEMENT IS STILL SUSPICIOUS
   If everyone converged on the same answer after Round 2, they may still share a blind spot.
   Verify the consensus answer just as rigorously as disputed ones.

3. JUDGE FLIP-FLOPPERS CAREFULLY
   If a model changed their answer R1→R2, examine why:
   • GOOD change: A real bug/flaw was identified with clear reasoning → trust the new answer
   • BAD change: Model just agreed with others without real justification → this is sycophancy, discount this model's input

4. IGNORE NON-ANSWERS
   If any model failed to produce an answer (API error, refusal, etc.), disregard it entirely.

5. USE CONFIDENCE SIGNALS
   Pay attention to confidence levels from earlier rounds. A model that said "Low confidence" and was later proven wrong is less surprising than one that said "High confidence" and failed.

6. YOU MUST VERIFY BEFORE RULING
   Never output a final answer you haven't personally verified.

STEP 1 - SURVEY:
Quickly assess:
• How many distinct candidate answers exist?
• Who agrees with whom?
• Who changed their answer R1→R2? Was it justified with real reasoning?
• What confidence levels were claimed?
• Any models that failed to answer? (Ignore them)

STEP 2 - VERIFY EACH CANDIDATE:
For each distinct proposed answer, test it yourself:

For CODE:
→ Trace Example 1 step-by-step, showing variable values at each line
→ Compare your output to expected output
→ CRITICAL: If any arrays were sorted, verify related data stayed paired
→ Test one edge case (n=0, n=1, negatives, large values)

For MATH:
→ Redo the key calculations yourself
→ Check magnitudes, signs, units
→ If data was reordered at any point, verify pairings preserved

For WRITING/ANALYSIS:
→ State the core thesis in one sentence
→ Does each paragraph support it? Is each claim backed by evidence?
→ What's the strongest counterargument? Is it addressed?

For ALL TASKS:
→ Does this actually answer what was asked?

Mark each candidate: ✓ VERIFIED or ✗ FAILED (state specific reason)

STEP 3 - RULE:
Based on your verification:
• One candidate verified, others failed → Select the verified one
• Multiple candidates verified → Select the best-reasoned or most complete
• ALL candidates failed → Construct the correct answer yourself with full work shown
• Genuinely uncertain → State what's unresolved, give your best judgment with caveats

STEP 4 - FINAL CHECK:
Before submitting, verify your final answer one more time:
→ Code: Trace Example 1. Does output match expected?
→ Math: Redo the key calculation. Does it hold?
→ Writing: Read paragraph first-sentences in sequence. Is the thesis supported?

If this check fails → return to Step 3 and fix.

OUTPUT FORMAT:
ANALYSIS:
[Summary: What candidates exist, who agreed/disagreed, who flipped (and whether justified), your verification result for each candidate with ✓/✗ and reasoning]

FINAL_ANSWER:
[The complete, correct answer]

JUSTIFICATION:
[Why this answer is correct — include your verification trace as proof. Why alternatives were rejected. Your confidence level (High/Medium/Low) and any remaining uncertainties.]`

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

  // For multi-line content (like code), extract everything until the next marker or end
  // Look for common markers that might come after
  const nextMarkers = [
    '\n\nRATIONALE:', '\n\nREASONING:', '\n\nFINAL_ANSWER:', '\n\nCRITIQUE:', '\n\nUPDATED_FINAL_ANSWER:',
    '\n\nJUSTIFICATION:', '\n\nANALYSIS:', '\n\nUNDERSTANDING:', '\n\nAPPROACH:', '\n\nSOLUTION:',
    '\n\nVERIFICATION:', '\n\nCONFIDENCE:', '\n\nSOLUTIONS_REVIEWED:', '\n\nBREAK_ATTEMPTS:',
    '\n\nBLIND_SPOT_CHECK:'
  ]
  let endIdx = restOfText.length

  for (const nextMarker of nextMarkers) {
    const markerIdx = restOfText.indexOf(nextMarker)
    if (markerIdx !== -1 && markerIdx < endIdx) {
      endIdx = markerIdx
    }
  }

  return restOfText.substring(0, endIdx).trim()
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
  maxTokens: number = 8192
): Promise<string> {
  try {
    // Cap max tokens based on model limits
    const getMaxTokensForProvider = (provider: string, requested: number): number => {
      const limits: Record<string, number> = {
        'openai': 4096,    // GPT-3.5-turbo max
        'anthropic': 4096, // Claude 3 Haiku max
        'google': 8192     // Gemini 2.5 Flash max
      }
      return Math.min(requested, limits[provider] || 4096)
    }

    const cappedMaxTokens = getMaxTokensForProvider(config.provider, maxTokens)

    switch (config.provider) {
      case 'anthropic': {
        const response = await clients.anthropic.messages.create({
          model: config.modelName,
          max_tokens: cappedMaxTokens,
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
          max_tokens: cappedMaxTokens,
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
            maxOutputTokens: cappedMaxTokens
          }
        })
        const result = await model.generateContent(userPrompt)

        // Check for safety filter blocks or incomplete responses
        const response = result.response
        const candidate = response.candidates?.[0]

        if (candidate?.finishReason === 'MAX_TOKENS') {
          console.warn(`Gemini hit MAX_TOKENS limit. Increase maxTokens if needed.`)
        }

        try {
          const text = response.text()
          if (!text || text.trim() === '') {
            console.warn(`Gemini returned empty response. Candidates:`, response.candidates)
            console.warn(`Prompt feedback:`, response.promptFeedback)
            return 'REASONING: Unable to generate response due to safety filters or API limitations.\n\nFINAL_ANSWER: No answer provided\n\nCRITIQUE: Unable to provide critique due to API error.\n\nUPDATED_FINAL_ANSWER: No answer provided'
          }
          return text
        } catch (e) {
          console.warn(`Gemini response.text() failed:`, e)
          return 'REASONING: Unable to generate response due to safety filters or API limitations.\n\nFINAL_ANSWER: No answer provided\n\nCRITIQUE: Unable to provide critique due to API error.\n\nUPDATED_FINAL_ANSWER: No answer provided'
        }
      }

      default:
        throw new Error(`Unknown provider: ${config.provider}`)
    }
  } catch (error) {
    console.error(`Error calling ${config.id}:`, error)

    // For Gemini errors, return a fallback response instead of crashing
    if (config.provider === 'google') {
      console.warn(`Gemini failed, returning fallback response`)
      // Return a response with all possible markers for different rounds
      return 'REASONING: API call failed.\n\nFINAL_ANSWER: No answer provided\n\nCRITIQUE: Unable to provide critique due to API error.\n\nUPDATED_FINAL_ANSWER: No answer provided'
    }

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
  const othersText = others
    .map(
      (o) => `ANSWER_FROM_${o.modelId}:
REASONING:
${o.reasoning}

FINAL_ANSWER:
${o.finalAnswer}`
    )
    .join('\n\n')

  return `QUESTION:
${question}

YOUR_PREVIOUS_ANSWER (${self.modelId}):
REASONING:
${self.reasoning}

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
${r1.reasoning}
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
${r2.critique}
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

  const responseText = await callLLM(ARBITER_MODEL, R3_SYSTEM_PROMPT, arbiterPrompt, clients, 8192)

  const finalAnswer = extractAfter(responseText, 'FINAL_ANSWER:')
  // Try JUSTIFICATION first (new format), fall back to RATIONALE (old format)
  let rationale = extractAfter(responseText, 'JUSTIFICATION:')
  if (!rationale) {
    rationale = extractAfter(responseText, 'RATIONALE:')
  }

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
