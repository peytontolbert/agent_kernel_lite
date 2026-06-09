export function normalizeF5TTSSpeechText(input) {
  let text = String(input || '').replace(/\s+/g, ' ').trim();
  if (!text) return text;

  const replacements = [
    [/\bF5\s*-?\s*TTS\b/gi, 'F five T T S'],
    [/\bF5TTS\b/gi, 'F five T T S'],
    [/\bF5\b/g, 'F five'],
    [/\bTTS\b/g, 'T T S'],
    [/\bWebGPU\b/g, 'Web G P U'],
    [/\bGPU\b/g, 'G P U'],
    [/\bWASM\b/g, 'Web Assembly'],
    [/\bWebAssembly\b/g, 'Web Assembly'],
    [/\bint4\b/gi, 'four bit'],
    [/\bq4\b/gi, 'four bit'],
    [/\b24\s*kHz\b/gi, 'twenty four kilohertz'],
    [/\b24\s*khz\b/gi, 'twenty four kilohertz'],
    [/\bkHz\b/g, 'kilohertz'],
    [/\bVocos\b/g, 'Voh coes'],
  ];
  for (const [pattern, replacement] of replacements) {
    text = text.replace(pattern, replacement);
  }
  return text.replace(/\s+/g, ' ').trim();
}
