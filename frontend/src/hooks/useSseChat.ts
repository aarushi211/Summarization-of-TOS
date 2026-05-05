/**
 * useSseChat.ts
 * React hook for the chat SSE stream.
 * Handles: token streaming, source parsing, typed error events, abort on unmount.
 */

"use client";

import { useState, useCallback, useRef } from "react";
import {
    openSseStream,
    consumeSseStream,
    CitedSource,
    SseEvent,
} from "@/lib/api";

export interface ChatMessage {
    role: "user" | "assistant";
    content: string;
    citedSources?: CitedSource[];
    error?: string;        // set if this message is an error
    isStreaming?: boolean;
}

export interface UseSseChatReturn {
    messages: ChatMessage[];
    isLoading: boolean;
    error: string | null;
    sendMessage: (query: string, documentId: string, serviceName?: string) => Promise<void>;
    clearError: () => void;
    clearChat: () => void;
}

export function useSseChat(): UseSseChatReturn {
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const abortRef = useRef<AbortController | null>(null);

    const clearError = useCallback(() => setError(null), []);
    const clearChat = useCallback(() => setMessages([]), []);

    const sendMessage = useCallback(async (
        query: string,
        documentId: string,
        serviceName: string = "Unknown Service",
    ) => {
        if (isLoading) return;

        // Cancel any in-flight stream
        abortRef.current?.abort();
        abortRef.current = new AbortController();

        setIsLoading(true);
        setError(null);

        // Append user message immediately
        setMessages(prev => [...prev, { role: "user", content: query }]);

        // Append placeholder assistant message (will be updated token-by-token)
        setMessages(prev => [
            ...prev,
            { role: "assistant", content: "", isStreaming: true },
        ]);

        try {
            const response = await openSseStream("/chat/stream", "POST", {
                query,
                document_id: documentId,
                service_name: serviceName,
            });

            if (!response) {
                // openSseStream redirected to login — nothing to do
                return;
            }

            await consumeSseStream(response, (event: SseEvent) => {
                switch (event.type) {
                    case "token":
                        // Append token to the streaming assistant message
                        setMessages(prev => {
                            const updated = [...prev];
                            const last = updated[updated.length - 1];
                            if (last?.role === "assistant") {
                                updated[updated.length - 1] = {
                                    ...last,
                                    content: last.content + event.data,
                                };
                            }
                            return updated;
                        });
                        break;

                    case "sources":
                        // Attach cited sources to the assistant message
                        setMessages(prev => {
                            const updated = [...prev];
                            const last = updated[updated.length - 1];
                            if (last?.role === "assistant") {
                                updated[updated.length - 1] = {
                                    ...last,
                                    citedSources: event.data,
                                };
                            }
                            return updated;
                        });
                        break;

                    case "done":
                        // Mark streaming complete
                        setMessages(prev => {
                            const updated = [...prev];
                            const last = updated[updated.length - 1];
                            if (last?.role === "assistant") {
                                updated[updated.length - 1] = {
                                    ...last,
                                    isStreaming: false,
                                };
                            }
                            return updated;
                        });
                        break;

                    case "error":
                        // Replace the streaming placeholder with an error message
                        setMessages(prev => {
                            const updated = [...prev];
                            const last = updated[updated.length - 1];
                            if (last?.role === "assistant") {
                                updated[updated.length - 1] = {
                                    ...last,
                                    isStreaming: false,
                                    error: event.data,
                                    // Keep any partial content that arrived before the error
                                    content: last.content || "",
                                };
                            }
                            return updated;
                        });
                        // Also surface at page level for prominent display
                        setError(event.data);
                        break;
                }
            });

        } catch (err) {
            const msg = err instanceof Error ? err.message : "Connection failed.";
            setMessages(prev => {
                const updated = [...prev];
                const last = updated[updated.length - 1];
                if (last?.role === "assistant") {
                    updated[updated.length - 1] = {
                        ...last,
                        isStreaming: false,
                        error: msg,
                    };
                }
                return updated;
            });
            setError(msg);
        } finally {
            setIsLoading(false);
        }
    }, [isLoading]);

    return { messages, isLoading, error, sendMessage, clearError, clearChat };
}