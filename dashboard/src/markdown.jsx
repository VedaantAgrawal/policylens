// Minimal **bold** rendering for chat answers — not a full markdown parser,
// just the one construct the model actually produces in this project's
// generation/agent prompts. Pulling in a markdown library for one pattern
// wasn't worth the dependency weight.
export function renderBold(text) {
  const parts = text.split(/(\*\*[^*]+\*\*)/g)
  return parts.map((part, i) => {
    if (part.startsWith('**') && part.endsWith('**')) {
      return <strong key={i}>{part.slice(2, -2)}</strong>
    }
    return part
  })
}
