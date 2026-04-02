import type { FC } from "react";
import { PersonaplexGithubLink } from "./PersonaplexGithubLink";

/**
 * Static About content (placeholder copy). Sections use print page-break hints and screen dividers.
 */
export const AboutTab: FC = () => {
  return (
    <div className="min-h-0 flex-1 overflow-y-auto">
      <article className="mx-auto max-w-2xl px-4 py-8 pb-16 text-white/90 sm:px-6 md:py-10">
        <header className="mb-10 border-b border-white/10 pb-8">
          <h1 className="text-xl font-semibold tracking-tight text-white sm:text-2xl">About</h1>
          <p className="mt-2 text-sm text-white/50">Overview, expectations, and how to use this prototype safely.</p>
        </header>

        <section
          className="space-y-4 border-b border-white/10 pb-12 print:break-after-page"
          aria-labelledby="about-guide"
        >
          <h2 id="about-guide" className="text-sm font-semibold uppercase tracking-[0.15em] text-white/55">
            Guide
          </h2>
          <p className="text-[0.95rem] leading-relaxed text-white/80">
            This is placeholder text for the guide section. Here you will find how to navigate the app, start a journal
            entry, use the knowledge base, and get the most out of Manual Journal Mode and AI-Assisted Journal Mode—written
            for new visitors and returning users alike.
          </p>
          <p className="text-[0.95rem] leading-relaxed text-white/80">
            Additional paragraphs can document keyboard shortcuts, import and export, and where your data lives. Replace
            this filler when you are ready to ship real documentation.
          </p>
        </section>

        <section
          className="mt-12 space-y-4 border-b border-white/10 pb-12 print:break-after-page"
          aria-labelledby="about-safety"
        >
          <h2 id="about-safety" className="text-sm font-semibold uppercase tracking-[0.15em] text-white/55">
            Safety
          </h2>
          <p className="text-[0.95rem] leading-relaxed text-white/80">
            Placeholder for privacy and safety expectations: what is sent to model providers, retention, and why you
            should avoid pasting highly sensitive secrets into a prototype environment.
          </p>
          <p className="text-[0.95rem] leading-relaxed text-white/80">
            A future version might link to a full privacy policy, data deletion instructions, and crisis resources where
            appropriate.
          </p>
        </section>

        <section className="mt-12 space-y-4 pb-8 print:break-after-page" aria-labelledby="about-purpose">
          <h2 id="about-purpose" className="text-sm font-semibold uppercase tracking-[0.15em] text-white/55">
            Purpose
          </h2>
          <p className="text-[0.95rem] leading-relaxed text-white/80">
            Placeholder for mission and intent: why Selfmeridian exists, who it is for, and how reflective journaling,
            learning, and memory-assisted chat fit together in one workspace.
          </p>
          <p className="text-[0.95rem] leading-relaxed text-white/80">
            Replace this with your product story, values, and any acknowledgements or open-source credits you want to
            highlight.
          </p>
        </section>

        <footer className="mt-16 border-t border-white/10 pt-10 pb-8" aria-label="Repository">
          <PersonaplexGithubLink className="pointer-events-auto" />
        </footer>
      </article>
    </div>
  );
};
