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

export async function fetchEvaluations(baseUrl) {
    const evaluations = (await (await fetch(baseUrl + '/__index__.json')).json())
    const evaluationsMap = new Map(evaluations.map(evaluation => [evaluation.id, evaluation]))
    return evaluationsMap
}

export async function fetchFiles(baseUrl, evaluations, benchmarkName, filePath) {
    const results = await Promise.all(Array.from(evaluations.values())
            .filter(evaluation => evaluation.benchmarks.includes(benchmarkName) || benchmarkName === 'total')
            .map(async modelInformation => {
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

export function getModelNumParams(modelInformation) {
    if (modelInformation.num_params)
        return modelInformation.num_params
    const modelName = modelInformation.model_name
    const match = modelName.match(/[0-9]+(B|b)/)
    if (match !== null)
        return match[0].toUpperCase()
    return ''
}
