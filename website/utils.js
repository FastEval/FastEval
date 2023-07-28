export function computeUpdatedHash(newItems) {
    const items = parseHash()
    for (const [k, v] of Object.entries(newItems)) {
        if (v === null && items.has(k))
            items.delete(k)
        else
            items.set(k, v)
    }

    return '?' + [...items.entries()].map(([k, v]) => k + '=' + v).join('&')
}

export function parseHash() {
    return new Map(Array.from(new URLSearchParams(location.hash.substring(1)).entries()))
}

export function round(num) {
    return num.toFixed(2)
}

export function allowCharacterLineBreaks(text, characters = ['/', '_']) {
    let out = text
    for (const char of characters)
        out = out.replaceAll(char, char + '\u200b')
    return out
}

export async function fetchEvaluations(baseUrl) {
    return (await (await fetch(baseUrl + '/__index__.json')).json())
}

export async function fetchFiles(baseUrl, index, benchmarkName, filePath) {
    const results = await Promise.all(index.filter(model => model.benchmarks.includes(benchmarkName)).map(async modelInformation => {
        const id = modelInformation.id
        const modelName = modelInformation.model_name
        const path = baseUrl
            + '/' + benchmarkName
            + '/' + modelName.replace('/', '--')
            + '/' + id
            + '/' + filePath
        const result = await (await fetch(path)).json()
        return [id, result]
    }))

    return new Map(results)
}

export function createEvaluationsMap(evaluations) {
    return new Map(evaluations.map(evaluation => [evaluation.id, evaluation]))
}

export function getModelNumParams(modelInformation) {
    if (modelInformation.num_params)
        return modelInformation.num_params
    const modelName = modelInformation.model_name
    const match = modelName.match(/[0-9]+(B|b)/)
    if (match !== null)
        return match[0].toUpperCase()
    return ''
}
