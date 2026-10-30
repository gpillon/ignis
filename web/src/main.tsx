import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "@fontsource/chakra-petch/500.css";
import "@fontsource/chakra-petch/600.css";
import "@fontsource-variable/instrument-sans/index.css";
import "@fontsource-variable/jetbrains-mono/index.css";
import { App } from "./app/App.tsx";
import { askForPersistence } from "./store/persist.ts";
import "./styles.css";

// Once per page load; a refusal changes nothing, the page saves as before.
void askForPersistence(navigator.storage);

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
