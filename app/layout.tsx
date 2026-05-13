import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "HAM10000 Light ML Pipeline",
  description: "Client-side HAM10000 subset training pipeline for Vercel deployment."
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
