import { NextResponse } from "next/server";

export function GET() {
  return NextResponse.json({
    ok: true,
    app: "HAM10000 Vercel Light Pipeline",
    backend: "Next.js Route Handler",
    training: "client-side TensorFlow.js"
  });
}
