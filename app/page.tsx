"use client"

import type React from "react"

import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Card } from "@/components/ui/card"
import { useState } from "react"

export default function TestPage() {
  const [question, setQuestion] = useState("")
  const [response, setResponse] = useState("")
  const [isLoading, setIsLoading] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!question.trim()) return

    setIsLoading(true)
    setResponse("")

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: [{ role: "user", content: question }],
        }),
      })

      if (!res.ok) throw new Error("API request failed")

      const reader = res.body?.getReader()
      const decoder = new TextDecoder()

      if (reader) {
        while (true) {
          const { done, value } = await reader.read()
          if (done) break

          const chunk = decoder.decode(value, { stream: true })
          setResponse((prev) => prev + chunk)
        }
      }
    } catch (error) {
      setResponse(`Error: ${error instanceof Error ? error.message : "Unknown error"}`)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <main className="min-h-screen bg-background p-6">
      <div className="max-w-4xl mx-auto space-y-6">
        <div>
          <h1 className="text-3xl font-bold">Multi-Agent API Tester</h1>
          <p className="text-muted-foreground mt-2">
            Test your multi-agent verification system (Claude → GPT-3.5 → Gemini)
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
                placeholder="e.g., If I invest $12,000 at 3% annual interest for 25 years, how much will I earn?"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                rows={4}
                disabled={isLoading}
              />
            </div>
            <Button type="submit" disabled={isLoading} className="w-full">
              {isLoading ? "Processing..." : "Test API"}
            </Button>
          </form>
        </Card>

        {response && (
          <Card className="p-6">
            <h2 className="text-lg font-semibold mb-3">Response</h2>
            <div className="prose prose-sm max-w-none whitespace-pre-wrap text-foreground">{response}</div>
          </Card>
        )}
      </div>
    </main>
  )
}
