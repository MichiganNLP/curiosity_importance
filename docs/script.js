const questionSets = {
  curious: [
    {
      label: "CAUSAL / EVIDENTIAL",
      text: "What evidence will strengthen this claim?",
      note: "Seeks evidence or an explanation that can close a meaningful knowledge gap.",
    },
    {
      label: "REFLECTIVE",
      text: "What assumption are you making here?",
      note: "Invites the user to examine evidence, uncertainty, or assumptions.",
    },
    {
      label: "EXPLORATORY",
      text: "What alternative explanation might fit this case?",
      note: "Broadens the space of possible interpretations or directions.",
    },
  ],
  flow: [
    {
      label: "ENGAGEMENT",
      text: "Would you like me to continue?",
      note: "Maintains the interaction without opening a new epistemic direction.",
    },
    {
      label: "TASK MANAGEMENT",
      text: "What format should this be in?",
      note: "Coordinates output requirements so the model can complete the task.",
    },
    {
      label: "CONTEXT REQUEST",
      text: "Can you share more context?",
      note: "Collects task information rather than inviting reflection or exploration.",
    },
  ],
};

const taxonomy = {
  information: {
    label: "INFORMATION GAP",
    example: "“What problem does it solve?”",
    description: "Surfaces missing knowledge or information needed to understand the issue.",
    theory: "Information-gap theory · epistemic curiosity",
    symbol: "?",
  },
  causal: {
    label: "CAUSAL–EXPLANATORY",
    example: "“Why did this happen?”",
    description: "Seeks reasons, mechanisms, or explanations rather than a surface description.",
    theory: "Explanation-seeking · epistemic curiosity",
    symbol: "↳",
  },
  exploratory: {
    label: "EXPLORATORY BROADENING",
    example: "“What else might explain this?”",
    description: "Opens additional interpretations, implications, or directions for inquiry.",
    theory: "Epistemic exploration · diversive curiosity",
    symbol: "+",
  },
  reflective: {
    label: "REFLECTIVE–METACOGNITIVE",
    example: "“What assumptions are you making?”",
    description: "Prompts examination of one's own evidence, uncertainty, or reasoning process.",
    theory: "Metacognition · uncertainty monitoring",
    symbol: "↺",
  },
  perspective: {
    label: "PERSPECTIVE TAKING",
    example: "“How might they interpret this?”",
    description: "Invites consideration of another person, role, or point of view.",
    theory: "Perspective-taking · role-taking theory",
    symbol: "⇄",
  },
  counterfactual: {
    label: "COUNTERFACTUAL–HYPOTHETICAL",
    example: "“What if the context changed?”",
    description: "Explores how an outcome might change under an imagined alternative condition.",
    theory: "Counterfactual reasoning · epistemic simulation",
    symbol: "≠",
  },
};

const header = document.querySelector("[data-header]");
const menuToggle = document.querySelector("[data-menu-toggle]");
const nav = document.querySelector("[data-nav]");

const updateHeader = () => {
  header?.classList.toggle("is-scrolled", window.scrollY > 16);
};

updateHeader();
window.addEventListener("scroll", updateHeader, { passive: true });

menuToggle?.addEventListener("click", () => {
  const willOpen = !nav.classList.contains("is-open");
  nav.classList.toggle("is-open", willOpen);
  document.body.classList.toggle("menu-open", willOpen);
  menuToggle.setAttribute("aria-expanded", String(willOpen));
});

nav?.querySelectorAll("a").forEach((link) => {
  link.addEventListener("click", () => {
    nav.classList.remove("is-open");
    document.body.classList.remove("menu-open");
    menuToggle?.setAttribute("aria-expanded", "false");
  });
});

let questionMode = "curious";
let questionIndex = 1;

const questionOutput = document.querySelector("[data-question-output]");
const questionLabel = document.querySelector("[data-question-label]");
const questionNote = document.querySelector("[data-question-note]");
const modeButtons = [...document.querySelectorAll("[data-question-mode]")];

function renderQuestion() {
  const examples = questionSets[questionMode];
  const example = examples[questionIndex % examples.length];
  questionOutput.textContent = example.text;
  questionLabel.textContent = example.label;
  questionNote.textContent = example.note;
}

modeButtons.forEach((button) => {
  button.addEventListener("click", () => {
    questionMode = button.dataset.questionMode;
    questionIndex = questionMode === "curious" ? 1 : 0;
    modeButtons.forEach((item) => {
      const selected = item === button;
      item.classList.toggle("is-active", selected);
      item.setAttribute("aria-selected", String(selected));
    });
    renderQuestion();
  });
});

document.querySelector("[data-shuffle]")?.addEventListener("click", () => {
  questionIndex = (questionIndex + 1) % questionSets[questionMode].length;
  renderQuestion();
});

const taxonomyTabs = [...document.querySelectorAll("[data-taxonomy]")];
const taxonomyFields = {
  label: document.querySelector("[data-taxonomy-label]"),
  example: document.querySelector("[data-taxonomy-example]"),
  description: document.querySelector("[data-taxonomy-description]"),
  theory: document.querySelector("[data-taxonomy-theory]"),
  symbol: document.querySelector("[data-taxonomy-symbol]"),
};

function selectTaxonomy(tab) {
  const item = taxonomy[tab.dataset.taxonomy];
  if (!item) return;

  taxonomyTabs.forEach((candidate) => {
    const selected = candidate === tab;
    candidate.classList.toggle("is-active", selected);
    candidate.setAttribute("aria-selected", String(selected));
    candidate.tabIndex = selected ? 0 : -1;
  });

  Object.entries(taxonomyFields).forEach(([key, node]) => {
    if (node) node.textContent = item[key];
  });
}

taxonomyTabs.forEach((tab, index) => {
  tab.addEventListener("click", () => selectTaxonomy(tab));
  tab.addEventListener("keydown", (event) => {
    if (!["ArrowDown", "ArrowUp", "ArrowRight", "ArrowLeft"].includes(event.key)) return;
    event.preventDefault();
    const forward = event.key === "ArrowDown" || event.key === "ArrowRight";
    const nextIndex = (index + (forward ? 1 : -1) + taxonomyTabs.length) % taxonomyTabs.length;
    taxonomyTabs[nextIndex].focus();
    selectTaxonomy(taxonomyTabs[nextIndex]);
  });
});

const copyButton = document.querySelector("[data-copy-citation]");

copyButton?.addEventListener("click", async () => {
  const citation = document.querySelector("#bibtex")?.textContent.trim();
  if (!citation) return;

  try {
    await navigator.clipboard.writeText(citation);
  } catch {
    const textarea = document.createElement("textarea");
    textarea.value = citation;
    textarea.setAttribute("readonly", "");
    textarea.style.position = "fixed";
    textarea.style.opacity = "0";
    document.body.appendChild(textarea);
    textarea.select();
    document.execCommand("copy");
    textarea.remove();
  }

  const initialText = copyButton.textContent;
  copyButton.textContent = "Copied!";
  window.setTimeout(() => {
    copyButton.textContent = initialText;
  }, 1800);
});

const revealItems = document.querySelectorAll(".reveal");

if ("IntersectionObserver" in window) {
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        entry.target.classList.add("is-visible");
        observer.unobserve(entry.target);
      });
    },
    { threshold: 0.12 },
  );

  revealItems.forEach((item) => observer.observe(item));
} else {
  revealItems.forEach((item) => item.classList.add("is-visible"));
}
