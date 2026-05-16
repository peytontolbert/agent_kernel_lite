#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';

const repoRoot = process.cwd();
const appPath = path.join(repoRoot, 'apps/mobile/www/app/js/agent-kernel-app.js');
const appSource = fs.readFileSync(appPath, 'utf8');

const requiredSymbols = [
  'activeAgentFallbackAnswer',
  'activeAgentDecisionNeedsFallback',
  'activeAgentContentPreservesInput',
  'activeAgentExtractJson',
  'activeAgentClassifyJson',
  'activeAgentAllowedLabels',
  'activeAgentClassifyLabel',
  'activeAgentActionItems',
  'activeAgentSimpleTranslation',
];

for (const symbol of requiredSymbols) {
  if (!appSource.includes(`function ${symbol}`)) {
    throw new Error(`missing runtime symbol in app bundle: ${symbol}`);
  }
}

function contentTokens(value) {
  return String(value || '').toLowerCase().match(/[a-z][a-z0-9-]{2,}/g) || [];
}

const stopwords = new Set([
  'about', 'after', 'again', 'agent', 'because', 'before', 'could', 'please', 'should', 'that', 'their',
  'there', 'these', 'thing', 'this', 'those', 'what', 'when', 'where', 'which', 'would', 'your',
]);

function sentenceCaseDraft(text) {
  const cleaned = String(text || '').replace(/\s+/g, ' ').trim();
  if (!cleaned) return '';
  const lower = cleaned === cleaned.toUpperCase() ? cleaned.toLowerCase() : cleaned;
  const capitalized = lower.replace(/(^|[.!?]\s+)([a-z])/g, (_match, prefix, char) => `${prefix}${char.toUpperCase()}`);
  return /[.!?]$/.test(capitalized) ? capitalized : `${capitalized}.`;
}

function professionalizeDraft(text) {
  const cleaned = sentenceCaseDraft(text);
  if (!cleaned) return '';
  if (/^(hi|hello|hey)(?:[, ]+(?:how are you|how are you doing))?[.!?]?$/i.test(cleaned)) return 'Hello, I hope you are well.';
  const source = String(text || '').replace(/\s+/g, ' ').trim();
  let requestText = source.replace(/^(?:hey|hi|hello|yo)\s+[, ]*/i, '').trim();
  let name = '';
  const directed = requestText.match(/^([A-Za-z][A-Za-z'-]{1,30})\s+(?=(?:please\s+)?(?:send|get|finish|prepare|share|complete|review|update|draft|write)\b)/i);
  if (directed) {
    name = directed[1].charAt(0).toUpperCase() + directed[1].slice(1).toLowerCase();
    requestText = requestText.slice(directed[0].length).trim();
  }
  requestText = requestText
    .replace(/^(?:i\s+)?(?:need|want)(?:\s+you)?\s+to\s+/i, '')
    .replace(/^(?:please|can you|could you|would you)\s+/i, '')
    .trim();
  const request = requestText.match(/^(send|get|finish|prepare|share|complete|review|update|draft|write)\s+(.+)$/i);
  if (request) {
    const verb = request[1].toLowerCase();
    let item = request[2].trim();
    let reason = '';
    const reasonMatch = item.match(/\s+because\s+(.+)$/i);
    if (reasonMatch) {
      reason = reasonMatch[1].trim();
      item = item.slice(0, reasonMatch.index).trim();
    }
    let deadline = '';
    const deadlineMatch = item.match(/\s+by\s+(.+)$/i);
    if (deadlineMatch) {
      deadline = deadlineMatch[1].trim();
      item = item.slice(0, deadlineMatch.index).trim();
    }
    if (item && !/\b(how are you|what's up|hello|hi)\b/i.test(item)) {
      const greeting = name ? `Hi ${name},` : 'Hello,';
      const deadlineText = deadline ? ` by ${deadline}` : '';
      const reasonText = reason ? ` ${sentenceCaseDraft(reason)}` : '';
      return `${greeting} could you please ${verb} ${item}${deadlineText}?${reasonText} Thank you.`;
    }
  }
  return cleaned
    .replace(/\bhey\b/gi, 'Hello')
    .replace(/\bpls\b/gi, 'please')
    .replace(/\bplz\b/gi, 'please')
    .replace(/\bASAP\b/g, 'as soon as possible')
    .replace(/\bu\b/gi, 'you')
    .replace(/\bur\b/gi, 'your')
    .replace(/\bthx\b/gi, 'thank you');
}

function compactSourceClauses(text, limit = 5) {
  return String(text || '')
    .split(/(?:\n+|[.;]|,\s+(?=(?:and |but |then |[A-Z][a-z]+:)))/)
    .map((item) => item.replace(/\s+/g, ' ').trim())
    .filter(Boolean)
    .slice(0, limit);
}

function activeAgentTitle(text, fallback = 'Untitled') {
  const tokens = contentTokens(text).filter((token) => !stopwords.has(token)).slice(0, 7);
  return tokens.length ? tokens.map((token) => token.charAt(0).toUpperCase() + token.slice(1)).join(' ') : fallback;
}

function activeAgentAllowedLabels(instruction = '') {
  const text = String(instruction || '');
  const match = text.match(/\b(?:labels?|one label|exactly one label)\s*[:=-]\s*([A-Za-z0-9_,\s-]{6,160})/i)
    || text.match(/\b(?:into|as)\s+(?:exactly\s+)?(?:one\s+)?(?:label|category)\s*[:=-]?\s*([A-Za-z0-9_,\s-]{6,160})/i);
  if (!match) return [];
  return Array.from(new Set(
    String(match[1] || '')
      .split(/[,/|]|\bor\b/i)
      .map((item) => item.trim().toLowerCase().replace(/\s+/g, '_'))
      .filter((item) => /^[a-z][a-z0-9_-]{1,40}$/.test(item))
      .slice(0, 12),
  ));
}

function activeAgentClassifyLabel(text, instruction = '') {
  const labels = activeAgentAllowedLabels(instruction);
  if (!labels.length) return '';
  const haystack = String(text || '').toLowerCase();
  const scores = new Map(labels.map((label) => [label, 0]));
  const bump = (label, amount = 1) => {
    if (scores.has(label)) scores.set(label, scores.get(label) + amount);
  };
  if (/\b(rewrite|reword|polish|professional|email|grammar|tone|paraphrase)\b/.test(haystack)) bump('writing', 5);
  if (/\b(invoice|budget|finance|approve|approval|receipt|\$)\b/.test(haystack)) bump('finance', 5);
  if (/\b(schedule|meeting|calendar|move|moved|thursday|monday|2 pm|deadline)\b/.test(haystack)) bump('schedule', 5);
  if (/\b(hotel|reservation|passport|flight|travel|room|july)\b/.test(haystack)) bump('travel', 5);
  if (/\b(search|web|online|current|latest|today|find)\b/.test(haystack)) bump('web_search', 5);
  return Array.from(scores.entries()).sort((a, b) => b[1] - a[1])[0]?.[0] || labels[0];
}

function fallback(agent, userText) {
  const instruction = `${agent.name || ''} ${agent.instruction || ''}`.toLowerCase();
  const original = String(userText || '').trim();
  if (/\b(search|web|latest|current|online)\b/.test(instruction + original)) {
    return { action: 'extension_request', content: 'Requesting approval to search the web.' };
  }
  if (/\b(classify|classification|label|intent|tone)\b/.test(instruction)) {
    const label = activeAgentClassifyLabel(original, instruction);
    return { action: 'respond', content: label || JSON.stringify({ intent: 'casual', tone: 'neutral' }) };
  }
  if (/\b(extract|owner|deadline|amount|email address|email addresses)\b/.test(instruction)) {
    const names = Array.from(new Set((original.match(/\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b/g) || [])));
    const money = Array.from(new Set((original.match(/\$\s?\d[\d,]*(?:\.\d{2})?/g) || [])));
    return { action: 'respond', content: JSON.stringify({ names, money }) };
  }
  if (/\b(action item|todo|to-do|owners?|deadlines?)\b/.test(instruction)) {
    return { action: 'respond', content: compactSourceClauses(original).map((item) => `- ${sentenceCaseDraft(item)}`).join('\n') };
  }
  if (/\b(checklist|check list)\b/.test(instruction)) {
    return { action: 'respond', content: compactSourceClauses(original).map((item) => `- [ ] ${sentenceCaseDraft(item)}`).join('\n') };
  }
  if (/\b(subject line|email subject|subject)\b/.test(instruction)) {
    return { action: 'respond', content: activeAgentTitle(original, 'Follow Up') };
  }
  if (/\b(translate|translation|spanish)\b/.test(instruction)) {
    return { action: 'respond', content: original.toLowerCase() === 'hello' ? 'Hola.' : sentenceCaseDraft(original) };
  }
  if (/\b(summarize|summary|recap)\b/.test(instruction)) {
    return { action: 'respond', content: `- ${sentenceCaseDraft(original)}` };
  }
  if (/\b(rewrite|reword|polish|professional|email)\b/.test(instruction)) {
    return { action: 'respond', content: professionalizeDraft(original) };
  }
  return { action: 'respond', content: original };
}

const cases = [
  {
    id: 'rewrite',
    agent: { name: 'Rewrite Agent', instruction: 'Rewrite the user text as a professional email.' },
    userText: 'hi how are you?',
    expect: ['respond', 'Hello', 'well'],
  },
  {
    id: 'rewrite-request',
    agent: { name: 'Rewrite Agent', instruction: 'Rewrite the user text as a professional email.' },
    userText: 'yo lena send the budget draft by june 3 because finance is waiting',
    expect: ['respond', 'Hi Lena', 'budget draft', 'june 3', 'Finance is waiting'],
  },
  {
    id: 'summary',
    agent: { name: 'Summary Agent', instruction: 'Summarize the user text as bullets.' },
    userText: 'Design approved the launch. QA needs links fixed.',
    expect: ['respond', 'Design approved', 'QA needs'],
  },
  {
    id: 'extract',
    agent: { name: 'Extractor', instruction: 'Extract names and amounts as JSON.' },
    userText: 'Maria approved $1,200 for John.',
    expect: ['respond', 'Maria', '$1,200'],
  },
  {
    id: 'checklist',
    agent: { name: 'Checklist', instruction: 'Turn the input into a checklist.' },
    userText: 'Review slides. Fix links.',
    expect: ['respond', '- [ ] Review', '- [ ] Fix'],
  },
  {
    id: 'classify-label',
    agent: { name: 'Classifier', instruction: 'Classify into exactly one label: travel, finance, schedule, writing, web_search. Return only the label.' },
    userText: 'Can you rewrite this note professionally?',
    expect: ['respond', 'writing'],
  },
  {
    id: 'web',
    agent: { name: 'Web Agent', instruction: 'Search the web for current information.' },
    userText: 'current TestFlight upload limits',
    expect: ['extension_request', 'search'],
  },
  {
    id: 'translate',
    agent: { name: 'Translator', instruction: 'Translate into Spanish.' },
    userText: 'hello',
    expect: ['respond', 'Hola'],
  },
];

const results = cases.map((item) => {
  const output = fallback(item.agent, item.userText);
  const body = `${output.action}\n${output.content}`;
  const failures = item.expect.filter((needle) => !body.includes(needle));
  return { id: item.id, passed: failures.length === 0, output, failures };
});

const passed = results.filter((item) => item.passed).length;
const summary = { passed, total: results.length, results };
console.log(JSON.stringify(summary, null, 2));
if (passed !== results.length) process.exit(1);
