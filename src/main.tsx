import React from "react";
import { createRoot } from "react-dom/client";
import PipelineApp from "../components/PipelineApp";
import "./styles.css";

createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <PipelineApp />
  </React.StrictMode>
);
