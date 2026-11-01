import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { Segmented } from "./Segmented.tsx";

describe("Segmented", () => {
  it("is the containing block of its hidden legend, so the legend stays inside the area that scrolls it", () => {
    // `sr-only` is `position: absolute`: with no positioned ancestor its box is the page's, which a scrolling
    // container does not clip, and a row below the fold made the whole page scroll (the Server view's scope).
    const html = renderToStaticMarkup(<Segmented legend="Scope" legendHidden name="scope" value="a" options={[{ value: "a", label: "A" }]} onChange={() => {}} />);
    expect(html).toMatch(/^<fieldset class="relative[ "]/);
    expect(html).toContain('<legend class="sr-only">Scope</legend>');
  });
});
