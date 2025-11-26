"use client"

import type React from "react"

import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Card } from "@/components/ui/card"
import { useState } from "react"

type ModelId = 'claude' | 'gpt' | 'gemini'

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

export default function TestPage() {
  const [question, setQuestion] = useState("")
  const [result, setResult] = useState<DebateResult | null>(null)
  const [error, setError] = useState("")
  const [isLoading, setIsLoading] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!question.trim()) return

    setIsLoading(true)
    setResult(null)
    setError("")

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: question.trim(),
          debug: true
        }),
      })

      if (!res.ok) {
        const errorData = await res.json()
        throw new Error(errorData.error || "API request failed")
      }

      const data: DebateResult = await res.json()
      setResult(data)
    } catch (err) {
      setError(`Error: ${err instanceof Error ? err.message : "Unknown error"}`)
    } finally {
      setIsLoading(false)
    }
  }

  const getModelName = (id: ModelId): string => {
    const names = {
      claude: "Claude (Haiku)",
      gpt: "GPT-4o Mini",
      gemini: "Gemini Flash"
    }
    return names[id]
  }

  const getModelColor = (id: ModelId): string => {
    const colors = {
      claude: "bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200",
      gpt: "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200",
      gemini: "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200"
    }
    return colors[id]
  }

  return (
    <main className="min-h-screen bg-background p-6">
      <div className="max-w-6xl mx-auto space-y-6">
        <div>
          <h1 className="text-3xl font-bold">Multi-Model Debate System</h1>
          <p className="text-muted-foreground mt-2">
            Watch 3 AI models debate and reach consensus through structured rounds
          </p>
        </div>

        <Card className="p-6">
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label htmlFor="question" className="text-sm font-medium block mb-2">
                Your Question
              </label>
              <Textarea
                id="question"
                placeholder="e.g., What is 15% of 240? or If I invest $12,000 at 3% annual interest for 25 years, how much will I earn?"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                rows={4}
                disabled={isLoading}
              />
            </div>
            <Button type="submit" disabled={isLoading} className="w-full">
              {isLoading ? "Running Debate..." : "Start Debate"}
            </Button>
          </form>
        </Card>

        {error && (
          <Card className="p-6 border-red-200 bg-red-50 dark:border-red-900 dark:bg-red-950">
            <p className="text-red-800 dark:text-red-200">{error}</p>
          </Card>
        )}

        {result && (
          <div className="space-y-6">
            {/* Final Answer */}
            <Card className="p-6 border-2 border-green-500 bg-green-50 dark:bg-green-950">
              <h2 className="text-xl font-bold mb-3 text-green-900 dark:text-green-100">
                Final Answer
              </h2>
              <p className="text-lg font-semibold mb-2 text-green-800 dark:text-green-200">
                {result.finalAnswer}
              </p>
              <p className="text-sm text-green-700 dark:text-green-300 mt-3">
                <strong>Rationale:</strong> {result.finalRationale}
              </p>
            </Card>

            {/* Round 1: Independent Solutions */}
            <Card className="p-6">
              <h2 className="text-xl font-bold mb-4">Round 1: Independent Solutions</h2>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {result.round1.map((answer) => (
                  <Card key={answer.modelId} className="p-4 border-2">
                    <div className="flex items-center gap-2 mb-3">
                      <span className={`px-2 py-1 rounded text-xs font-semibold ${getModelColor(answer.modelId)}`}>
                        {getModelName(answer.modelId)}
                      </span>
                    </div>
                    <div className="space-y-3">
                      <div>
                        <p className="text-xs font-semibold text-muted-foreground mb-1">REASONING:</p>
                        <p className="text-sm whitespace-pre-wrap line-clamp-4">
                          {answer.reasoning.substring(0, 200)}...
                        </p>
                      </div>
                      <div>
                        <p className="text-xs font-semibold text-muted-foreground mb-1">FINAL ANSWER:</p>
                        <p className="text-sm font-semibold">{answer.finalAnswer}</p>
                      </div>
                    </div>
                  </Card>
                ))}
              </div>
            </Card>

            {/* Round 2: Critique & Revise */}
            <Card className="p-6">
              <h2 className="text-xl font-bold mb-4">Round 2: Critique & Revise</h2>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {result.round2.map((critique) => (
                  <Card key={critique.modelId} className="p-4 border-2">
                    <div className="flex items-center gap-2 mb-3">
                      <span className={`px-2 py-1 rounded text-xs font-semibold ${getModelColor(critique.modelId)}`}>
                        {getModelName(critique.modelId)}
                      </span>
                    </div>
                    <div className="space-y-3">
                      <div>
                        <p className="text-xs font-semibold text-muted-foreground mb-1">CRITIQUE:</p>
                        <p className="text-sm whitespace-pre-wrap line-clamp-3">
                          {critique.critique.substring(0, 150)}...
                        </p>
                      </div>
                      <div>
                        <p className="text-xs font-semibold text-muted-foreground mb-1">UPDATED ANSWER:</p>
                        <p className="text-sm font-semibold">{critique.revisedAnswer}</p>
                      </div>
                    </div>
                  </Card>
                ))}
              </div>
            </Card>

            {/* Full Details (Collapsible) */}
            <details className="group">
              <summary className="cursor-pointer text-sm font-medium text-muted-foreground hover:text-foreground">
                View Full Debate Transcript
              </summary>
              <Card className="mt-4 p-6">
                <pre className="text-xs whitespace-pre-wrap overflow-auto max-h-96">
                  {JSON.stringify(result, null, 2)}
                </pre>
              </Card>
            </details>
          </div>
        )}
      </div>
    </main>
  )
}
