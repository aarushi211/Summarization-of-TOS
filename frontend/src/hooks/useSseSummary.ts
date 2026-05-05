/**
 * useSseSummary.ts
 * React hook for the per-topic summary SSE stream.
 * Topics arrive one at a time as they complete on the server.
 */

"use client";

import { useState, useCallback } from "react";
import {
    openSseStream,
    consumeSseStream,
    TopicResult,
    SseEvent,
} from "@/lib/api";

export interface UseSseSummaryReturn {
    topics: TopicResult[];
    isLoading: boolean;
    isDone: boolean;
    error: string | null;
    generateSummary: (
        documentId: string,
        serviceName: string,
        docType: string,
    ) => Promise<void>;
    clearSummary: () => void;
}

export function useSseSummary(): UseSseSummaryReturn {
    const [topics, setTopics] = useState<TopicResult[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [isDone, setIsDone] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const clearSummary = useCallback(() => {
        setTopics([]);
        setIsDone(false);
        setError(null);
    }, []);

    const generateSummary = useCallback(async (
        documentId: string,
        serviceName: string,
        docType: string,
    ) => {
        if (isLoading) return;

        setIsLoading(true);
        setIsDone(false);
        setError(null);
        setTopics([]);

        const params = new URLSearchParams({
            document_id: documentId,
            service_name: serviceName,
            doc_type: docType,
        });

        try {
            const response = await openSseStream(
                `/summary/stream?${params.toString()}`,
                "GET",
            );
            if (!response) return;

            await consumeSseStream(response, (event: SseEvent) => {
                switch (event.type) {
                    case "topic_ready":
                        // Append topic as it arrives — UI can render incrementally
                        setTopics(prev => [...prev, event.data]);
                        break;

                    case "done":
                        setIsDone(true);
                        break;

                    case "error":
                        // Fatal stream error — surface it and stop
                        setError(event.data);
                        setIsDone(true);
                        break;

                    // topic_ready errors (per-topic retrieval failure) are embedded in
                    // the TopicResult.error field — handled by the rendering layer
                }
            });

        } catch (err) {
            setError(err instanceof Error ? err.message : "Failed to generate summary.");
            setIsDone(true);
        } finally {
            setIsLoading(false);
        }
    }, [isLoading]);

    return { topics, isLoading, isDone, error, generateSummary, clearSummary };
}