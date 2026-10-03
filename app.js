const STORAGE_KEYS = {
  examples: "octo-ai-training-examples-v1",
  documents: "octo-ai-local-documents-v1",
};

const BUILT_IN_DOCUMENTS = [
  {
    id: "prototype-overview",
    title: "프로토타입 안내",
    url: "#prototype",
    text: "이 프로토타입은 한국어로 질문을 입력하면 브라우저에 준비된 자료와 사용자가 추가한 자료에서 관련 내용을 검색합니다. 검색된 자료의 문장을 근거로 보여주며, 관련 자료가 없으면 모른다고 답합니다. 현재 생성형 언어 모델은 연결되어 있지 않아 자유로운 문장 생성이나 일반 지식 답변은 제공하지 않습니다.",
  },
  {
    id: "training-status",
    title: "모델 및 훈련 상태",
    url: "#training",
    text: "현재 0.6B 모델의 가중치, 토크나이저, 실제 학습 엔진은 준비되어 있지 않습니다. 훈련 데이터 화면에서는 한국어 대화 예시를 브라우저 로컬 저장소에 저장하고 JSON 파일로 가져오거나 내보낼 수 있습니다. 이 화면은 실제 모델을 훈련하지 않으며 진행률을 시뮬레이션하지 않습니다.",
  },
  {
    id: "data-and-sources",
    title: "자료 및 출처 사용",
    url: "#sources",
    text: "답변은 저장된 자료에서 찾은 내용에만 근거해야 합니다. 자료를 찾지 못하거나 질문과 자료의 관련성이 낮으면 추측하지 않고 모른다고 답합니다. 출처 표시는 답변 근거로 사용된 문서의 제목을 제공합니다. 사용자는 훈련 데이터 탭에서 별도의 한국어 예시를 관리할 수 있습니다.",
  },
];

const conversation = document.querySelector("#conversation");
const chatForm = document.querySelector("#chat-form");
const chatInput = document.querySelector("#chat-input");
const datasetForm = document.querySelector("#dataset-form");
const promptInput = document.querySelector("#example-prompt");
const responseInput = document.querySelector("#example-response");
const exampleList = document.querySelector("#example-list");
const datasetEmpty = document.querySelector("#dataset-empty");
const toast = document.querySelector("#toast");
let toastTimer;

function readStoredArray(key) {
  try {
    const parsed = JSON.parse(localStorage.getItem(key) || "[]");
    return Array.isArray(parsed) ? parsed : [];
  } catch (error) {
    console.error(`Could not read ${key} from local storage.`, error);
    return [];
  }
}

function saveArray(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
    return true;
  } catch (error) {
    console.error(`Could not save ${key} to local storage.`, error);
    showToast("브라우저 저장 공간에 저장하지 못했습니다.");
    return false;
  }
}

function showToast(message) {
  toast.textContent = message;
  toast.classList.add("show");
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => toast.classList.remove("show"), 2800);
}

function createId() {
  if (typeof crypto.randomUUID === "function") return crypto.randomUUID();
  return `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function makeMessage(role, text, sources = []) {
  const article = document.createElement("article");
  article.className = `message ${role}`;

  if (role === "assistant") {
    const avatar = document.createElement("span");
    avatar.className = "assistant-avatar";
    avatar.setAttribute("aria-hidden", "true");
    avatar.textContent = "o";
    article.append(avatar);
  }

  const body = document.createElement("div");
  body.className = "message-body";
  const label = document.createElement("p");
  label.className = "message-label";
  label.textContent = role === "assistant" ? "Octo · 검색 기반 답변" : "나";
  const content = document.createElement("p");
  content.className = "message-text";
  content.textContent = text;
  body.append(label, content);

  if (sources.length) {
    const citationList = document.createElement("div");
    citationList.className = "citation-list";
    citationList.setAttribute("aria-label", "답변 출처");
    sources.forEach((source, index) => {
      const citation = document.createElement("a");
      citation.className = "citation";
      citation.href = source.url;
      if (source.url.startsWith("#")) {
        citation.addEventListener("click", event => event.preventDefault());
      } else {
        citation.target = "_blank";
        citation.rel = "noopener noreferrer";
      }
      const number = document.createElement("span");
      number.className = "citation-number";
      number.textContent = `[${index + 1}]`;
      const title = document.createElement("span");
      title.textContent = source.title;
      citation.append(number, title);
      citationList.append(citation);
    });
    body.append(citationList);
  }

  if (role === "assistant") {
    const note = document.createElement("span");
    note.className = "answer-note";
    note.textContent = "검색된 자료를 바탕으로 한 프로토타입 응답";
    body.append(note);
  }

  article.append(body);
  conversation.append(article);
  article.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function normalizeText(text) {
  return text.toLocaleLowerCase("ko-KR").replace(/[^\p{L}\p{N}\s]/gu, " ").replace(/\s+/g, " ").trim();
}

function textTerms(text) {
  const words = normalizeText(text).split(" ").filter(word => word.length > 1);
  const terms = new Set(words);
  for (const word of words) {
    if (word.length > 2) {
      for (let index = 0; index < word.length - 1; index += 1) {
        terms.add(word.slice(index, index + 2));
      }
    }
  }
  return terms;
}

function getDocuments() {
  const customDocuments = readStoredArray(STORAGE_KEYS.documents);
  return [...BUILT_IN_DOCUMENTS, ...customDocuments];
}

function searchDocuments(query) {
  const queryTerms = textTerms(query);
  if (!queryTerms.size) return [];

  return getDocuments()
    .map(document => {
      const content = `${document.title} ${document.text}`;
      const documentTerms = textTerms(content);
      let matched = 0;
      for (const term of queryTerms) {
        if (documentTerms.has(term)) matched += 1;
      }
      const overlap = matched / queryTerms.size;
      const phraseBonus = normalizeText(document.text).includes(normalizeText(query)) ? 0.5 : 0;
      return { document, score: overlap + phraseBonus };
    })
    .filter(result => result.score >= 0.12)
    .sort((left, right) => right.score - left.score)
    .slice(0, 3)
    .map(result => result.document);
}

function answerFromSources(query, sources) {
  if (!sources.length) {
    return {
      text: "현재 저장된 자료에서는 답을 찾지 못했어요. 근거 없이 추측하지 않겠습니다. 관련 문서를 훈련 데이터 탭에서 추가한 뒤 다시 질문해 주세요.",
      sources: [],
    };
  }

  const queryTerms = textTerms(query);
  const excerpts = [];
  const citedSources = [];
  for (const source of sources) {
    const sentences = source.text.split(/(?<=[.!?。])\s+|(?<=니다\.)\s*/u).map(sentence => sentence.trim()).filter(Boolean);
    const bestSentence = sentences
      .map(sentence => {
        const terms = textTerms(sentence);
        let matches = 0;
        for (const term of queryTerms) if (terms.has(term)) matches += 1;
        return { sentence, score: matches / Math.max(queryTerms.size, 1) };
      })
      .sort((left, right) => right.score - left.score)[0];
    if (bestSentence && bestSentence.score > 0) {
      excerpts.push(bestSentence.sentence);
      citedSources.push(source);
    }
  }

  if (!excerpts.length) {
    return {
      text: "관련된 자료는 찾았지만 질문에 답할 만큼 직접적인 근거는 확인하지 못했어요. 더 구체적인 질문이나 관련 문서를 추가해 주세요.",
      sources: [],
    };
  }
  return { text: excerpts.join("\n\n"), sources: citedSources };
}

function submitQuestion(question) {
  const trimmed = question.trim();
  if (!trimmed) return;
  makeMessage("user", trimmed);
  chatInput.value = "";
  chatInput.style.height = "auto";

  const result = answerFromSources(trimmed, searchDocuments(trimmed));
  makeMessage("assistant", result.text, result.sources);
}

chatForm.addEventListener("submit", event => {
  event.preventDefault();
  submitQuestion(chatInput.value);
});

chatInput.addEventListener("input", () => {
  chatInput.style.height = "auto";
  chatInput.style.height = `${Math.min(chatInput.scrollHeight, 150)}px`;
});

chatInput.addEventListener("keydown", event => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    chatForm.requestSubmit();
  }
});

document.querySelectorAll(".suggestion").forEach(button => {
  button.addEventListener("click", () => submitQuestion(button.dataset.prompt || ""));
});

document.querySelector("#clear-chat").addEventListener("click", () => {
  conversation.replaceChildren();
  showToast("대화를 지웠습니다.");
});

document.querySelectorAll(".nav-item").forEach(button => {
  button.addEventListener("click", () => {
    const selectedTab = button.dataset.tab;
    const isChat = selectedTab === "chat";
    document.querySelectorAll(".nav-item").forEach(navItem => {
      navItem.classList.toggle("active", navItem === button);
      if (navItem === button) navItem.setAttribute("aria-current", "page");
      else navItem.removeAttribute("aria-current");
    });
    document.querySelector("#chat-page").hidden = !isChat;
    document.querySelector("#training-page").hidden = isChat;
    document.querySelector("#page-title").textContent = isChat ? "대화" : "훈련 데이터";
    document.querySelector("#clear-chat").hidden = !isChat;
    if (!isChat) renderExamples();
  });
});

function renderExamples() {
  const examples = readStoredArray(STORAGE_KEYS.examples);
  exampleList.replaceChildren();
  document.querySelector("#dataset-count").textContent = `${examples.length}개`;
  datasetEmpty.hidden = examples.length > 0;
  examples.forEach((example, index) => {
    const card = document.createElement("article");
    card.className = "example-card";
    const copy = document.createElement("div");
    copy.className = "example-copy";
    const prompt = document.createElement("strong");
    prompt.textContent = `사용자: ${example.prompt}`;
    const response = document.createElement("p");
    response.textContent = `답변: ${example.response}`;
    copy.append(prompt, response);
    const remove = document.createElement("button");
    remove.className = "delete-example";
    remove.type = "button";
    remove.textContent = "삭제";
    remove.setAttribute("aria-label", `예시 ${index + 1} 삭제`);
    remove.addEventListener("click", () => {
      const current = readStoredArray(STORAGE_KEYS.examples);
      current.splice(index, 1);
      if (saveArray(STORAGE_KEYS.examples, current)) renderExamples();
    });
    card.append(copy, remove);
    exampleList.append(card);
  });
}

function renderDocuments() {
  const documents = readStoredArray(STORAGE_KEYS.documents);
  const list = document.querySelector("#document-list");
  list.replaceChildren();
  document.querySelector("#document-count").textContent = `내 자료 ${documents.length}개`;
  documents.forEach((source, index) => {
    const row = document.createElement("div");
    row.className = "document-row";
    const link = document.createElement("a");
    link.textContent = source.title;
    link.href = source.url || `#${source.id}`;
    if (source.url) {
      link.target = "_blank";
      link.rel = "noopener noreferrer";
    } else {
      link.addEventListener("click", event => event.preventDefault());
    }
    const preview = document.createElement("span");
    preview.textContent = source.text.length > 90 ? `${source.text.slice(0, 90)}…` : source.text;
    link.append(preview);
    const remove = document.createElement("button");
    remove.className = "delete-example";
    remove.type = "button";
    remove.textContent = "삭제";
    remove.setAttribute("aria-label", `${source.title} 자료 삭제`);
    remove.addEventListener("click", () => {
      const current = readStoredArray(STORAGE_KEYS.documents);
      current.splice(index, 1);
      if (saveArray(STORAGE_KEYS.documents, current)) {
        renderDocuments();
        showToast("검색 자료를 삭제했습니다.");
      }
    });
    row.append(link, remove);
    list.append(row);
  });
}

document.querySelector("#document-form").addEventListener("submit", event => {
  event.preventDefault();
  const title = document.querySelector("#document-title").value.trim();
  const urlValue = document.querySelector("#document-url").value.trim();
  const text = document.querySelector("#document-content").value.trim();
  if (!title || !text) return;

  let url = "";
  if (urlValue) {
    try {
      const parsed = new URL(urlValue);
      if (parsed.protocol !== "https:" && parsed.protocol !== "http:") throw new Error("Only HTTP(S) source links are allowed.");
      url = parsed.href;
    } catch (error) {
      showToast("출처 링크는 http 또는 https 주소여야 합니다.");
      return;
    }
  }

  const documents = readStoredArray(STORAGE_KEYS.documents);
  documents.unshift({ id: `user-${createId()}`, title, url, text });
  if (saveArray(STORAGE_KEYS.documents, documents)) {
    event.currentTarget.reset();
    renderDocuments();
    showToast("검색 자료를 추가했습니다.");
  }
});

datasetForm.addEventListener("submit", event => {
  event.preventDefault();
  const prompt = promptInput.value.trim();
  const response = responseInput.value.trim();
  if (!prompt || !response) return;
  const examples = readStoredArray(STORAGE_KEYS.examples);
  examples.unshift({ prompt, response, createdAt: new Date().toISOString() });
  if (saveArray(STORAGE_KEYS.examples, examples)) {
    datasetForm.reset();
    renderExamples();
    showToast("한국어 대화 예시를 저장했습니다.");
  }
});

document.querySelector("#export-data").addEventListener("click", () => {
  const examples = readStoredArray(STORAGE_KEYS.examples);
  const blob = new Blob([JSON.stringify({ format: "octo-ai-dataset-v1", language: "ko", examples }, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "octo-ai-training-data.json";
  link.click();
  URL.revokeObjectURL(url);
  showToast(`${examples.length}개 예시를 JSON으로 내보냈습니다.`);
});

document.querySelector("#import-data").addEventListener("click", () => document.querySelector("#import-file").click());
document.querySelector("#import-file").addEventListener("change", async event => {
  const [file] = event.target.files || [];
  if (!file) return;
  try {
    const imported = JSON.parse(await file.text());
    const examples = Array.isArray(imported) ? imported : imported.examples;
    if (!Array.isArray(examples) || examples.some(example => typeof example.prompt !== "string" || typeof example.response !== "string")) {
      throw new Error("Expected an array of prompt/response examples.");
    }
    const cleanExamples = examples.map(example => ({
      prompt: example.prompt.trim().slice(0, 4000),
      response: example.response.trim().slice(0, 8000),
      createdAt: typeof example.createdAt === "string" ? example.createdAt : new Date().toISOString(),
    })).filter(example => example.prompt && example.response);
    if (saveArray(STORAGE_KEYS.examples, [...cleanExamples, ...readStoredArray(STORAGE_KEYS.examples)])) {
      renderExamples();
      showToast(`${cleanExamples.length}개 예시를 가져왔습니다.`);
    }
  } catch (error) {
    console.error("Could not import training examples.", error);
    showToast("JSON 형식이 올바르지 않습니다.");
  } finally {
    event.target.value = "";
  }
});

function checkBrowserReadiness() {
  const supportsWebGPU = Boolean(navigator.gpu);
  document.querySelector("#webgpu-status").textContent = supportsWebGPU
    ? "이 브라우저에서 API를 사용할 수 있습니다"
    : "이 브라우저에서 지원하지 않습니다";
  document.querySelector("#webgpu-tag").textContent = supportsWebGPU ? "지원" : "미지원";
  document.querySelector("#webgpu-tag").classList.toggle("unsupported", !supportsWebGPU);

  const memory = navigator.deviceMemory;
  if (typeof memory === "number") {
    document.querySelector("#memory-status").textContent = `브라우저 보고값 약 ${memory} GB (시스템 전체 RAM과 다를 수 있음)`;
    document.querySelector("#memory-tag").textContent = "참고";
  } else {
    document.querySelector("#memory-status").textContent = "브라우저에서 메모리 정보를 제공하지 않습니다";
    document.querySelector("#memory-tag").textContent = "비공개";
  }
}

renderExamples();
renderDocuments();
checkBrowserReadiness();
makeMessage("assistant", "안녕하세요! 저는 지금 저장된 자료를 검색해 근거와 함께 답하는 초기 프로토타입이에요. 아직 생성형 AI 모델은 연결되어 있지 않으니, 답을 자료에서 찾지 못하면 모른다고 말씀드릴게요.");
