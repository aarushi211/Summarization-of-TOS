import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "TOS Summarizer | Legal AI",
  description: "AI-powered legal document analysis and summarization.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
