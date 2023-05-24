export function computeUpdatedHash(newItems) {
    const items = parseHash()
    for (const [k, v] of Object.entries(newItems))
        items.set(k, v)
    return '?' + [...items.entries()].map(([k, v]) => k + '=' + v).join('&')
}

export function parseHash() {
    return new Map(Array.from(new URLSearchParams(location.hash.substring(1)).entries()))
}

export function round(num) {
    return Math.round(num * 10000) / 10000
}

export function allowCharacterLineBreaks(text, characters = ['/', '_']) {
    let out = text
    for (const char of characters)
        out = out.replaceAll(char, char + '\u200b')
    return out
}
