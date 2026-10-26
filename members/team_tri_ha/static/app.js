
/* =========================================
   TRI-HA RESEARCH AI
   Frontend interactions and citation handling
   ========================================= */

// =========================================
// 1. HTML ELEMENTS
// =========================================

const questionInput = document.getElementById("question");
const askButton = document.getElementById("askButton");

const resultPanel = document.getElementById("resultPanel");
const findingsView = document.getElementById("findingsView");
const evidenceView = document.getElementById("evidenceView");

const answerBox = document.getElementById("answer");
const contextBox = document.getElementById("context");
const sourceCards = document.getElementById("sourceCards");

const copyButton = document.getElementById("copyButton");
const viewEvidenceButton = document.getElementById("viewEvidenceButton");
const backToAnswerButton = document.getElementById("backToAnswerButton");

// =========================================
// 2. VIEW SWITCHING
// =========================================

function showFindings() {
    findingsView.hidden = false;
    evidenceView.hidden = true;
    resultPanel.scrollTop = 0;
}

function showEvidence() {
    findingsView.hidden = true;
    evidenceView.hidden = false;
    resultPanel.scrollTop = 0;
}

// =========================================
// 3. PAPER NAME HELPERS
// =========================================

function formatPaperTitle(filename) {
    return String(filename || "")
        .replace(/\.pdf$/i, "")
        .replace(/^\d+_/, "")
        .replace(/_/g, " ")
        .trim();
}

function normalizePaperName(value) {
    return String(value || "")
        .toLowerCase()
        .replace(/\.pdf$/i, "")
        .replace(/^\d+_/, "")
        .replace(/[^a-z0-9]+/g, " ")
        .trim();
}

// Match the answer's paper label to the
// actual retrieved paper filename.

function paperNameMatches(label, filename) {
    const paper = normalizePaperName(filename);
    const name = normalizePaperName(label);

    if (!paper || !name) {
        return false;
    }

    if (paper === name) {
        return true;
    }

    const aliases = {
        "rag": [
            "retrieval augmented generation"
        ],
        "llama": [
            "llama"
        ],
        "llama 3": [
            "the llama 3 herd of models",
            "llama 3 herd of models"
        ],
        "deepseek v3": [
            "deepseek v3"
        ],
        "transformer": [
            "attention is all you need"
        ],
        "bert": [
            "bert"
        ]
    };

    const possibleNames = aliases[name] || [name];

    return possibleNames.some(alias => {
        const normalizedAlias = normalizePaperName(alias);

        return (
            paper === normalizedAlias ||
            paper.startsWith(normalizedAlias + " ")
        );
    });
}

// =========================================
// 4. CITATION PATTERNS
// =========================================

// Recognizes:
//
// [paper.pdf p.7]
// [file p.7]
// [19 p.1]
// [19]
//
// Unrecognized citations remain unchanged.

const allCitationPattern =
    /\[([^\]\r\n]+\.pdf)\s+p\.(\d+)\]|\[file\s+p\.(\d+)\]|\[(\d+)\s+p\.(\d+)\]|\[(\d+)\]/gi;

function sourceKey(filename, page) {
    return (
        normalizePaperName(filename) +
        "|" +
        String(Number(page))
    );
}

// =========================================
// 5. PARSE RETRIEVED EVIDENCE
// =========================================

function parseResearchContext(context) {
    const rawContext = Array.isArray(context)
        ? context.join("\n\n")
        : String(context || "");

    if (!rawContext.trim()) {
        return [];
    }

    const pattern =
        /\[([^\]\r\n]+\.pdf)\s+p\.(\d+)\][ \t]*/gi;

    const matches = [...rawContext.matchAll(pattern)];
    const passages = [];

    for (let i = 0; i < matches.length; i++) {
        const match = matches[i];

        const start = match.index + match[0].length;

        const end = i + 1 < matches.length
            ? matches[i + 1].index
            : rawContext.length;

        const passageText = rawContext
            .slice(start, end)
            .trim();

        if (!passageText) {
            continue;
        }

        passages.push({
            filename: match[1].trim(),
            page: Number(match[2]),
            text: passageText
        });
    }

    return passages;
}

// =========================================
// 6. CITATION LOOKUP
// =========================================

function buildCitationMap(passages) {
    const map = new Map();

    passages.forEach((passage, index) => {
        const key = sourceKey(
            passage.filename,
            passage.page
        );

        if (!map.has(key)) {
            map.set(key, index + 1);
        }
    });

    return map;
}

// Find the first passage with a matching
// paper name AND page number.

function findSourceByPaperAndPage(
    paperLabel,
    page,
    passages
) {
    const matches = [];

    passages.forEach((passage, index) => {
        if (
            Number(passage.page) === Number(page) &&
            paperNameMatches(
                paperLabel,
                passage.filename
            )
        ) {
            matches.push(index);
        }
    });

    if (matches.length === 0) {
        return null;
    }

    // Multiple passages from the same
    // paper and page are acceptable.
    // Navigate to the first matching card.

    const filenames = new Set(
        matches.map(index =>
            normalizePaperName(passages[index].filename)
        )
    );

    if (filenames.size !== 1) {
        return null;
    }

    return matches[0] + 1;
}

// =========================================
// 7. IDENTIFY THE PAPER FOR A CITATION
// =========================================

// Example:
//
// LLaMA: The A100 GPU was used [file p.4].
//
// The current line tells us the paper is LLaMA.

function getPaperLabelBeforeCitation(
    answer,
    citationPosition
) {
    const beforeCitation = answer.slice(
        0,
        citationPosition
    );

    const currentLine = beforeCitation
        .split(/\r?\n/)
        .pop()
        .trim();

    const match = currentLine.match(
        /^([^:\n]{1,100}):\s+/
    );

    if (!match) {
        return null;
    }

    return match[1].trim();
}

// =========================================
// 8. CREATE CLICKABLE CITATION
// =========================================

function createCitationLink(sourceNumber, passage) {
    const citation = document.createElement("a");

    citation.href = `#source-${sourceNumber}`;
    citation.className = "citation-link";
    citation.textContent = `[${sourceNumber}]`;

    citation.title =
        `${formatPaperTitle(passage.filename)}, page ${passage.page}`;

    citation.setAttribute(
        "aria-label",
        `View source ${sourceNumber}: ` +
        `${formatPaperTitle(passage.filename)}, ` +
        `page ${passage.page}`
    );

    citation.addEventListener("click", event => {
        event.preventDefault();
        navigateToSource(sourceNumber);
    });

    return citation;
}

// =========================================
// 9. RENDER ANSWER WITH CITATIONS
// =========================================

function renderResearchAnswer(answer, passages) {
    answerBox.replaceChildren();

    const rawAnswer = String(answer || "");

    if (!rawAnswer.trim()) {
        answerBox.textContent = "No answer was generated.";
        return;
    }

    const citationMap = buildCitationMap(passages);

    let lastIndex = 0;

    for (const match of rawAnswer.matchAll(
        allCitationPattern
    )) {
        answerBox.appendChild(
            document.createTextNode(
                rawAnswer.slice(
                    lastIndex,
                    match.index
                )
            )
        );

        let sourceNumber = null;

        // ---------------------------------
        // Case A: [paper.pdf p.7]
        // ---------------------------------

        if (match[1]) {
            const key = sourceKey(
                match[1],
                match[2]
            );

            sourceNumber =
                citationMap.get(key) || null;
        }

        // ---------------------------------
        // Case B: [file p.7]
        // ---------------------------------

        else if (match[3]) {
            const page = Number(match[3]);

            const paperLabel =
                getPaperLabelBeforeCitation(
                    rawAnswer,
                    match.index
                );

            if (paperLabel) {
                sourceNumber =
                    findSourceByPaperAndPage(
                        paperLabel,
                        page,
                        passages
                    );
            }

            // Fallback: link only if every
            // matching passage belongs to
            // the same paper.

            if (sourceNumber === null) {
                const matches = passages
                    .map((passage, index) => ({
                        passage,
                        index
                    }))
                    .filter(item =>
                        Number(item.passage.page) === page
                    );

                const filenames = new Set(
                    matches.map(item =>
                        normalizePaperName(
                            item.passage.filename
                        )
                    )
                );

                if (
                    matches.length > 0 &&
                    filenames.size === 1
                ) {
                    sourceNumber =
                        matches[0].index + 1;
                }
            }
        }

        // ---------------------------------
        // Case C: [19 p.1]
        // ---------------------------------

        else if (match[4] && match[5]) {
            const number = Number(match[4]);
            const page = Number(match[5]);

            const candidate =
                passages[number - 1];

            if (
                candidate &&
                Number(candidate.page) === page
            ) {
                sourceNumber = number;
            }
        }

        // ---------------------------------
        // Case D: [19]
        // ---------------------------------

        else if (match[6]) {
            const number = Number(match[6]);

            if (passages[number - 1]) {
                sourceNumber = number;
            }
        }

        // ---------------------------------
        // Render resolved citation
        // ---------------------------------

        if (
            sourceNumber !== null &&
            passages[sourceNumber - 1]
        ) {
            answerBox.appendChild(
                createCitationLink(
                    sourceNumber,
                    passages[sourceNumber - 1]
                )
            );
        } else {
            // Never invent a source.
            answerBox.appendChild(
                document.createTextNode(match[0])
            );
        }

        lastIndex =
            match.index + match[0].length;
    }

    answerBox.appendChild(
        document.createTextNode(
            rawAnswer.slice(lastIndex)
        )
    );
}

// =========================================
// 10. NAVIGATE TO EVIDENCE
// =========================================

function navigateToSource(sourceNumber) {
    showEvidence();

    const target = document.getElementById(
        `source-${sourceNumber}`
    );

    if (!target) {
        return;
    }

    target.open = true;

    document.querySelectorAll(
        ".source-highlight"
    ).forEach(card => {
        card.classList.remove(
            "source-highlight"
        );
    });

    target.classList.add(
        "source-highlight"
    );

    requestAnimationFrame(() => {
        target.scrollIntoView({
            behavior: "smooth",
            block: "center"
        });
    });

    setTimeout(() => {
        target.classList.remove(
            "source-highlight"
        );
    }, 1800);
}

// =========================================
// 11. RENDER EVIDENCE CARDS
// =========================================

function renderSourceCards(passages) {
    sourceCards.replaceChildren();

    if (passages.length === 0) {
        const message =
            document.createElement("p");

        message.className = "no-evidence";

        message.textContent =
            "No structured source passages " +
            "are available for this answer.";

        sourceCards.appendChild(message);
        return;
    }

    passages.forEach((passage, index) => {
        const sourceNumber = index + 1;

        const card =
            document.createElement("details");

        card.className = "source-card";
        card.id = `source-${sourceNumber}`;

        const heading =
            document.createElement("summary");

        heading.className = "source-heading";

        const sourceInfo =
            document.createElement("div");

        sourceInfo.className = "source-info";

        const numberLabel =
            document.createElement("span");

        numberLabel.className = "source-number";
        numberLabel.textContent =
            `SOURCE ${sourceNumber}`;

        const title =
            document.createElement("strong");

        title.className = "source-title";
        title.textContent =
            formatPaperTitle(
                passage.filename
            );

        const page =
            document.createElement("span");

        page.className = "source-page";
        page.textContent =
            `Page ${passage.page}`;

        sourceInfo.appendChild(
            numberLabel
        );

        sourceInfo.appendChild(title);

        heading.appendChild(
            sourceInfo
        );

        heading.appendChild(page);

        const preview =
            document.createElement("p");

        preview.className = "source-preview";

        preview.textContent =
            passage.text.slice(0, 180) +
            (passage.text.length > 180
                ? "…"
                : "");

        const fullText =
            document.createElement("p");

        fullText.className =
            "source-full-text";

        fullText.textContent =
            passage.text;

        card.appendChild(heading);
        card.appendChild(preview);
        card.appendChild(fullText);

        sourceCards.appendChild(card);
    });
}

// =========================================
// 12. SEND QUESTION TO FASTAPI
// =========================================

async function askResearchAI() {
    const question =
        questionInput.value.trim();

    if (!question) {
        alert(
            "Please enter a research question."
        );

        questionInput.focus();
        return;
    }

    if (askButton.disabled) {
        return;
    }

    askButton.disabled = true;
    askButton.textContent =
        "Searching research...";

    resultPanel.hidden = true;

    try {
        const response = await fetch(
            "/team1/rag-agent",
            {
                method: "POST",
                headers: {
                    "Content-Type":
                        "application/json"
                },
                body: JSON.stringify({
                    query: question
                })
            }
        );

        if (!response.ok) {
            throw new Error(
                `FastAPI returned HTTP ${response.status}`
            );
        }

        const data =
            await response.json();

        if (
            !data ||
            typeof data.answer !== "string" ||
            !(
                typeof data.context === "string" ||
                Array.isArray(data.context)
            )
        ) {
            throw new Error(
                "Unexpected API response."
            );
        }

        const passages =
            parseResearchContext(
                data.context
            );

        renderSourceCards(passages);

        renderResearchAnswer(
            data.answer,
            passages
        );

        contextBox.textContent =
            Array.isArray(data.context)
                ? data.context.join("\n\n")
                : data.context;

        copyButton.textContent =
            "Copy answer";

        showFindings();

        resultPanel.hidden = false;

    } catch (error) {
        console.error(
            "Research AI error:",
            error
        );

        alert(
            "Unable to get an answer.\n\n" +
            error.message
        );

    } finally {
        askButton.disabled = false;
        askButton.textContent =
            "Ask Research AI";
    }
}

// =========================================
// 13. BUTTON EVENTS
// =========================================

askButton.addEventListener(
    "click",
    askResearchAI
);

viewEvidenceButton.addEventListener(
    "click",
    showEvidence
);

backToAnswerButton.addEventListener(
    "click",
    showFindings
);

// =========================================
// 14. SUGGESTED QUESTIONS
// =========================================

document.querySelectorAll(
    ".suggestion-button"
).forEach(button => {
    button.addEventListener(
        "click",
        () => {
            const question =
                button.dataset.question;

            if (question) {
                questionInput.value =
                    question;

                questionInput.focus();
            }
        }
    );
});

// =========================================
// 15. COPY ANSWER
// =========================================

copyButton.addEventListener(
    "click",
    async () => {
        try {
            await navigator.clipboard.writeText(
                answerBox.textContent || ""
            );

            copyButton.textContent =
                "Copied!";

        } catch (error) {
            console.error(
                "Copy failed:",
                error
            );

            copyButton.textContent =
                "Copy failed";
        }
    }
);
