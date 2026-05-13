"use client";

import dynamic from "next/dynamic";

const PipelineApp = dynamic(() => import("./PipelineApp"), {
  ssr: false,
  loading: () => <main className="container"><section className="hero"><h1>Loading HAM10000 pipeline...</h1></section></main>
});

export default function ClientOnlyPipeline() {
  return <PipelineApp />;
}
