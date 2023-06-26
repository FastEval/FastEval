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
    return num.toFixed(2)
}

export function allowCharacterLineBreaks(text, characters = ['/', '_']) {
    let out = text
    for (const char of characters)
        out = out.replaceAll(char, char + '\u200b')
    return out
}

export async function fetchModels(baseUrl) {
    return (await (await fetch(baseUrl + '/__index__.json')).json())
}

export function fetchFiles(baseUrl, models, benchmarkName, end='.json') {
    return Promise.all(models
        .filter(model => model.benchmarks.includes(benchmarkName))
        .map(model => model.model_name)
        .map(async model => [model, await fetch(baseUrl + '/' + benchmarkName + '/' + model.replace('/', '--') + end).then(r => r.json())])
    )
}

export function createModelsMap(models) {
    return Object.fromEntries(models.map(({ model_name, ...rest }) => [model_name, { model_name: model_name, ...rest }]))
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
