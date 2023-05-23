import { parseHash } from './utils.js'
import { createAllBenchmarksV } from './benchmarks/main.js'
import * as OpenAIEvals from './benchmarks/openai-evals.js'
import * as Vicuna from './benchmarks/vicuna.js'

async function createSingleBenchmarkV(baseUrl, benchmarkName, parameters) {
    switch (benchmarkName) {
        case 'openai-evals':
            return await OpenAIEvals.createV(baseUrl, parameters)
        case 'vicuna':
            return await Vicuna.createV(baseUrl, parameters)
        default:
            throw new Error()
    }
}

async function createMainV() {
    window.addEventListener('hashchange', () => {
        location.reload()
    })

    const baseUrl = './reports'
    const hashParameters = parseHash()

    if (hashParameters.has('benchmark'))
        return createSingleBenchmarkV(baseUrl, hashParameters.get('benchmark'), hashParameters)
    return await createAllBenchmarksV(baseUrl)
}

createMainV().then(mainV => document.body.appendChild(mainV))
