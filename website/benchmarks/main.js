import { round, parseHash } from '../utils.js'
import { createLinkE } from '../components/link.js'
import { createExplanationTextE } from '../components/text.js'
import * as OpenAIEvals from '../benchmarks/openai-evals.js'
import * as Vicuna from '../benchmarks/vicuna.js'

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

export async function createBenchmarksIndexV(baseUrl) {
    const models = await (await fetch(baseUrl + '/__index__.json')).json()

    const [vicunaEvaluationResults, ...openaiEvalsResults] = await Promise.all([
        fetch(baseUrl + '/vicuna/reviews.json').then(r => r.json()),
        ...models.map(async model => [model, await fetch(baseUrl + '/openai-evals/' + model.replace('/', '--') + '/__index__.json').then(r => r.json())]),
    ])

    const relativeOpenAiEvalsScores = OpenAIEvals.computeRelativeOpenAiEvalsScores(Object.fromEntries(openaiEvalsResults)).averageRelativeScoresByModelName

    const tableE = document.createElement('table')
    const theadE = tableE.createTHead().insertRow()
    theadE.insertCell().appendChild(createExplanationTextE('Model'))
    theadE.insertCell().appendChild(createLinkE('OpenAI Evals Score', { benchmark: 'openai-evals' }))
    theadE.insertCell().appendChild(createLinkE('Vicuna Elo Rank', { benchmark: 'vicuna' }))
    theadE.insertCell().appendChild(createExplanationTextE('Vicuna Win Percentage'))
    const tbodyE = tableE.createTBody()
    for (const model of models) {
        const rowE = tbodyE.insertRow()
        rowE.insertCell().appendChild(createExplanationTextE(model))
        rowE.insertCell().appendChild(createExplanationTextE(round(relativeOpenAiEvalsScores[model])))

        const vicunaModelResults = vicunaEvaluationResults.models[model]
        rowE.insertCell().appendChild(createExplanationTextE(Math.round(vicunaModelResults.elo_rank)))
        const winRate = (vicunaModelResults.num_wins + vicunaModelResults.num_ties / 2) / vicunaModelResults.num_matches
        rowE.insertCell().appendChild(createExplanationTextE(round(winRate)))
    }

    return tableE
}

export async function createBenchmarksV(baseUrl) {
    const hashParameters = parseHash()
    if (hashParameters.has('benchmark'))
        return createSingleBenchmarkV(baseUrl, hashParameters.get('benchmark'), hashParameters)
    return await createBenchmarksIndexV(baseUrl)
}
