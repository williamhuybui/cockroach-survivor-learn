
/* =========================================
   TRI-HA RESEARCH AI
   Frontend interactions
   Compatible with Phase 2 and Phase 3 RAG
   ========================================= */

// 1. Get HTML elements

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


// 2. Switch between findings and evidence

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


// 3. Convert PDF filenames into readable titles

function formatPaperTitle(filename) {
    return String(filename || "")
        .replace(/\.pdf$/i, "")
        .replace(/^\d+_/, "")
        .replace(/_/g, " ")
        .trim();
}


// 4. Remove raw PDF citations from the visible answer

function cleanResearchAnswer(answer) {
    return String(answer || "")
        // Remove labels like [18_Mixtral_of_Experts.pdf p.1]
        .replace(/\[[^\]\r\n]+\.pdf\s+p\.\d+\]/gi, "")

        // Remove spaces before punctuation
        .replace(/[ \t]+([.,;!?])/g, "$1")

        // Normalize repeated spaces
        .replace(/[ \t]{2,}/g, " ")

        // Normalize excessive blank lines
        .replace(/\n{3,}/g, "\n\n")

        .trim();
}


// 5. Parse retrieved context into individual passages

function parseResearchContext(context) {
    // Accept either a string or an array of strings.
    // Phase 3 Agent.ask() originally returns an array,
    // while our FastAPI endpoint joins it into a string.

    const rawContext = Array.isArray(context)
        ? context.join("\n\n")
        : String(context || "");

    if (!rawContext.trim()) {
        return [];
    }

    // Supports BOTH formats:
    //
    // Phase 2:
    // [paper.pdf p.1]
    // Evidence text...
    //
    // Phase 3:
    // [paper.pdf p.1] Evidence text...

    const pattern = /\[([^\]\r\n]+\.pdf)\s+p\.(\d+)\][ \t]*/g;

    const matches = [...rawContext.matchAll(pattern)];
    const passages = [];

    for (let i = 0; i < matches.length; i++) {
        const match = matches[i];

        const start = match.index + match[0].length;

        const end = i + 1 < matches.length
            ? matches[i + 1].index
            : rawContext.length;

        let passageText = rawContext
            .slice(start, end)
            .trim();

        const pageNumber = match[2];

        // Remove a standalone trailing page number
        // only if it matches the source page.
        const lines = passageText.split(/\r?\n/);

        if (
            lines.length > 1 &&
            lines[lines.length - 1].trim() === pageNumber
        ) {
            lines.pop();
            passageText = lines.join("\n").trim();
        }

        // Ignore empty passages.
        if (!passageText) {
            continue;
        }

        passages.push({
            filename: match[1],
            page: pageNumber,
            text: passageText
        });
    }

    return passages;
}


// 6. Render supporting evidence as expandable cards

function renderSourceCards(context) {
    sourceCards.replaceChildren();

    const passages = parseResearchContext(context);

    if (passages.length === 0) {
        const message = document.createElement("p");

        message.className = "no-evidence";

        message.textContent =
            "No structured source passages are available for this answer.";

        sourceCards.appendChild(message);
        return;
    }

    passages.forEach((passage, index) => {
        const card = document.createElement("details");
        card.className = "source-card";

        const heading = document.createElement("summary");
        heading.className = "source-heading";

        const sourceInfo = document.createElement("div");
        sourceInfo.className = "source-info";

        const sourceNumber = document.createElement("span");
        sourceNumber.className = "source-number";
        sourceNumber.textContent = `SOURCE ${index + 1}`;

        const sourceTitle = document.createElement("strong");
        sourceTitle.className = "source-title";
        sourceTitle.textContent = formatPaperTitle(passage.filename);

        const pageLabel = document.createElement("span");
        pageLabel.className = "source-page";
        pageLabel.textContent = `Page ${passage.page}`;

        sourceInfo.appendChild(sourceNumber);
        sourceInfo.appendChild(sourceTitle);

        heading.appendChild(sourceInfo);
        heading.appendChild(pageLabel);

        const preview = document.createElement("p");
        preview.className = "source-preview";

        const previewText = passage.text.slice(0, 180);

        preview.textContent =
            previewText +
            (passage.text.length > 180 ? "…" : "");

        const fullText = document.createElement("p");
        fullText.className = "source-full-text";
        fullText.textContent = passage.text;

        card.appendChild(heading);
        card.appendChild(preview);
        card.appendChild(fullText);

        sourceCards.appendChild(card);
    });
}


// 7. Send the research question to FastAPI

async function askResearchAI() {
    const question = questionInput.value.trim();

    if (!question) {
        alert("Please enter a research question.");
        questionInput.focus();
        return;
    }

    // Prevent duplicate requests.
    if (askButton.disabled) {
        return;
    }

    askButton.disabled = true;
    askButton.textContent = "Searching research...";

    // Hide previous results while processing.
    resultPanel.hidden = true;

    try {
        const response = await fetch("/team1/rag-agent", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                query: question
            })
        });

        if (!response.ok) {
            let message = `FastAPI returned HTTP ${response.status}`;

            try {
                const errorData = await response.json();

                if (typeof errorData.detail === "string") {
                    message += `: ${errorData.detail}`;
                }
            } catch {
                // Keep the original HTTP error message.
            }

            throw new Error(message);
        }

        const data = await response.json();

        // Validate the expected API response.
        if (
            !data ||
            typeof data.answer !== "string" ||
            !(
                typeof data.context === "string" ||
                Array.isArray(data.context)
            )
        ) {
            throw new Error(
                "Unexpected API response. Expected answer and context."
            );
        }

        // Clean citations from the displayed answer.
        const cleanAnswer = cleanResearchAnswer(data.answer);

        answerBox.textContent =
            cleanAnswer || "No answer was generated.";

        // Preserve the original retrieved context.
        const contextText = Array.isArray(data.context)
            ? data.context.join("\n\n")
            : data.context;

        contextBox.textContent = contextText;

        // Create supporting evidence cards.
        renderSourceCards(contextText);

        // Reset the Copy button.
        copyButton.textContent = "Copy answer";

        // Show findings first.
        showFindings();

        resultPanel.hidden = false;

    } catch (error) {
        console.error("Research AI error:", error);

        alert(
            "Unable to get an answer. " +
            "Check that FastAPI and Ollama are running.\n\n" +
            error.message
        );

    } finally {
        askButton.disabled = false;
        askButton.textContent = "Ask Research AI";
    }
}


// 8. Connect main buttons

askButton.addEventListener("click", askResearchAI);

viewEvidenceButton.addEventListener("click", showEvidence);

backToAnswerButton.addEventListener("click", showFindings);


// 9. Suggested research questions

document.querySelectorAll(".suggestion-button").forEach(button => {
    button.addEventListener("click", () => {
        const suggestedQuestion = button.dataset.question;

        if (suggestedQuestion) {
            questionInput.value = suggestedQuestion;
            questionInput.focus();
        }
    });
});


// 10. Copy the cleaned research answer

copyButton.addEventListener("click", async () => {
    try {
        await navigator.clipboard.writeText(
            answerBox.textContent || ""
        );

        copyButton.textContent = "Copied!";

    } catch (error) {
        console.error("Copy failed:", error);
        copyButton.textContent = "Copy failed";
    }
});
