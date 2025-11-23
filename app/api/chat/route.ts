import { NextRequest } from "next/server"

import { GoogleGenerativeAI } from "@google/generative-ai"
import OpenAI from "openai"
import Anthropic from "@anthropic-ai/sdk"

type Message = {
  role: "user" | "assistant"
  content: string
}

export async function POST(req: NextRequest) {
  try {
    const { messages } = (await req.json()) as { messages: Message[] }

    if (!messages || messages.length === 0) {
      return new Response("Messages are required", { status: 400 })
    }

    const lastMessage = messages[messages.length - 1]
    if (!lastMessage || lastMessage.role !== "user") {
      return new Response("Last message must be from user", { status: 400 })
    }

    // Initialize AI clients
    const anthropic = new Anthropic({
      apiKey: process.env.ANTHROPIC_API_KEY,
    })

    const openai = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY,
    })

    const genAI = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY || "")

    const encoder = new TextEncoder()
    const stream = new ReadableStream({
      async start(controller) {
        try {
          const userMessage = lastMessage.content

          // AGENT 1: Initial Problem Solver
          const claudeResponse = await anthropic.messages.create({
            model: "claude-3-haiku-20240307",
            max_tokens: 2048,
            messages: [
              {
                role: "user",
                content: `You are Agent 1 - the Initial Problem Solver in a multi-agent verification system.

USER'S QUESTION:
${userMessage}

YOUR TASK:
1. Break down the problem into clear steps
2. Show ALL calculations explicitly with intermediate steps (e.g., "12,000 × 0.03 = 360, then 360 × 25 = 9,000")
3. State all assumptions clearly
4. If there are constraints, list them explicitly and check each one
5. For numerical problems: write out EVERY calculation in full, don't skip steps
6. If the problem involves multiple variables, create a clear structure (table/list) to track them
7. Double-check your own arithmetic before finalizing

CRITICAL: Your response will be verified by other agents, so accuracy is more important than speed. Show your work clearly so others can verify it.

Provide your detailed solution:`
              }
            ],
          })

          const claudeText = claudeResponse.content[0].type === 'text'
            ? claudeResponse.content[0].text
            : ''

          console.log("Agent 1 (Claude) response:", claudeText)

          // AGENT 2: Fact Checker & Verifier
          const gptResponse = await openai.chat.completions.create({
            model: "gpt-3.5-turbo",
            max_tokens: 2048,
            messages: [
              {
                role: "user",
                content: `You are Agent 2 - the Fact Checker & Verifier in a multi-agent verification system.

USER'S ORIGINAL QUESTION:
${userMessage}

AGENT 1'S SOLUTION:
${claudeText}

YOUR CRITICAL VERIFICATION TASK:
1. **VERIFY ALL CALCULATIONS**: Re-calculate every single number from scratch. Check:
   - Basic arithmetic (addition, subtraction, multiplication, division)
   - Order of operations
   - Unit conversions
   - Percentages and decimals
   
2. **CHECK LOGICAL CONSISTENCY**: 
   - Does the solution actually answer the question asked?
   - Are there any logical contradictions?
   - Are all constraints from the problem satisfied?
   
3. **IDENTIFY ERRORS**:
   - Mark each error with "❌ ERROR:" followed by explanation
   - For calculation errors, show: "Agent 1 calculated X, but correct answer is Y"
   - Note if Agent 1 misunderstood any constraints
   
4. **CONFIRM CORRECT PARTS**:
   - Mark verified correct sections with "✓ VERIFIED:"
   - This helps Agent 3 know what's trustworthy

5. **ADD MISSING INFORMATION**:
   - If Agent 1 missed important aspects, note them clearly

FORMAT YOUR RESPONSE AS:
## Verification Results

### Errors Found
[List all errors here, or write "No errors found"]

### Verified Correct
[List what you've verified as accurate]

### Additional Notes
[Any missing information or improvements]

BE THOROUGH - catching errors now prevents them from propagating to the final answer.`
              }
            ],
          })

          const gptText = gptResponse.choices[0]?.message?.content || ''
          console.log("Agent 2 (GPT-3.5) response:", gptText)

          // AGENT 3: Final Synthesizer & Quality Controller
          const geminiModel = genAI.getGenerativeModel({ model: "gemini-2.5-flash" })
          const geminiResponse = await geminiModel.generateContent({
            contents: [
              {
                role: "user",
                parts: [
                  {
                    text: `You are Agent 3 - the Final Synthesizer & Quality Controller in a multi-agent verification system.

USER'S ORIGINAL QUESTION:
${userMessage}

AGENT 1'S INITIAL SOLUTION:
${claudeText}

AGENT 2'S VERIFICATION & CORRECTIONS:
${gptText}

YOUR FINAL SYNTHESIS TASK:
1. **INDEPENDENT VERIFICATION**: Before synthesizing, independently verify any calculations that Agent 2 flagged as errors. Don't just trust Agent 2 - verify the corrections yourself.

2. **RESOLVE CONTRADICTIONS**:
   - If Agent 1 and Agent 2 disagree, determine which is correct by recalculating
   - Show your verification work
   - If both are wrong, provide the correct answer with calculations

3. **CREATE FINAL ANSWER**:
   - Use only verified correct information
   - Correct all identified errors
   - Present a clean, accurate final response
   - Include all necessary calculations clearly

4. **FINAL QUALITY CHECKS**:
   - Does the answer directly address the user's question?
   - Are all numbers accurate and properly calculated?
   - Are all constraints satisfied?
   - Is anything still missing?

5. **FLAG UNSOLVABLE PROBLEMS**: If the problem has no valid solution given the constraints, clearly state this rather than forcing an invalid answer.

FORMAT:
Start with a brief summary, then provide the complete verified answer. If you found errors in previous agents' work, briefly note what was corrected.

REMEMBER: You are the last line of defense against errors. The user trusts this final response to be accurate.`
                  }
                ]
              }
            ]
          })

          const geminiText = geminiResponse.response.text()
          controller.enqueue(encoder.encode(geminiText))

          controller.close()
        } catch (error) {
          console.error("Multi-agent error:", error)
          controller.enqueue(encoder.encode(`\n\nError: ${error instanceof Error ? error.message : 'Unknown error occurred'}`))
          controller.close()
        }
      },
    })

    return new Response(stream, {
      headers: {
        "Content-Type": "text/plain; charset=utf-8",
        "Transfer-Encoding": "chunked",
      },
    })
  } catch (error) {
    console.error("Chat API error:", error)
    return new Response("Internal server error", { status: 500 })
  }
}