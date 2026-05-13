type VercelResponse = {
  status: (code: number) => VercelResponse;
  json: (body: unknown) => void;
};

export default function handler(_req: unknown, res: VercelResponse) {
  res.status(200).json({
    ok: true,
    app: "HAM10000 Vercel Light Pipeline",
    frontend: "Vite + React + TypeScript",
    backend: "Vercel Serverless Function",
    training: "client-side TensorFlow.js"
  });
}
