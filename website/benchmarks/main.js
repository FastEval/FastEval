import { round, parseHash } from '../utils.js'
import { createLinkE } from '../components/link.js'
import { createTextE } from '../components/text.js'
import { createTableScoreCell } from '../components/table-score-cell.js'
import * as OpenAIEvals from '../benchmarks/openai-evals.js'
import * as Vicuna from '../benchmarks/vicuna.js'
import * as LMEvaluationHarness from '../benchmarks/lm-evaluation-harness.js'

async function createSingleBenchmarkV(baseUrl, benchmarkName, parameters) {
    switch (benchmarkName) {
        case 'openai-evals':
            return await OpenAIEvals.createV(baseUrl, parameters)
        case 'vicuna':
            return await Vicuna.createV(baseUrl, parameters)
        case 'lm-evaluation-harness':
            return await LMEvaluationHarness.createV(baseUrl)
        default:
            throw new Error()
    }
}

export async function createBenchmarksIndexV(baseUrl) {
    const models = (await (await fetch(baseUrl + '/__index__.json')).json())

    const [vicunaEvaluationResults, openaiEvalsResults, lmEvaluationHarnessResults] = await Promise.all([
        fetch(baseUrl + '/vicuna/reviews.json').then(r => r.json()),
        Promise.all(models.filter(model => model.benchmarks.includes('openai-evals')).map(model => model.model_name)
            .map(async model => [model, await fetch(baseUrl + '/openai-evals/' + model.replace('/', '--') + '/__index__.json').then(r => r.json())])),
        Promise.all(models.filter(model => model.benchmarks.includes('lm-evaluation-harness')).map(model => model.model_name)
            .map(async model => [model, await fetch(baseUrl + '/lm-evaluation-harness/' + model.replace('/', '--') + '.json').then(r => r.json())]))
    ])

    const relativeOpenAiEvalsScores = OpenAIEvals.computeRelativeOpenAiEvalsScores(Object.fromEntries(openaiEvalsResults)).averageRelativeScoresByModelName

    const averageLmEvaluationHarnessScores =  Object.fromEntries(lmEvaluationHarnessResults.map(([modelName, results]) =>
        [modelName, LMEvaluationHarness.computeAverageScore(results.results)]))

    const tableE = document.createElement('table')
    const theadE = tableE.createTHead().insertRow()
    theadE.insertCell().appendChild(createTextE('Model'))
    theadE.insertCell().appendChild(createLinkE('OpenAI Evals Score', { benchmark: 'openai-evals' }))
    theadE.insertCell().appendChild(createLinkE('Vicuna Elo Rank', { benchmark: 'vicuna' }))
    theadE.insertCell().appendChild(createTextE('Vicuna Win Percentage'))
    theadE.insertCell().appendChild(createLinkE('lm-evaluation-harness', { benchmark: 'lm-evaluation-harness' }))
    const tbodyE = tableE.createTBody()
    for (const { model_name: model, benchmarks } of models) {
        const rowE = tbodyE.insertRow()
        rowE.insertCell().appendChild(createTextE(model))

        if (benchmarks.includes('openai-evals'))
            createTableScoreCell(rowE, createTextE(round(relativeOpenAiEvalsScores[model])))
        else
            rowE.insertCell()

        if (benchmarks.includes('vicuna')) {
            const vicunaModelResults = vicunaEvaluationResults.models[model]
            createTableScoreCell(rowE, createTextE(Math.round(vicunaModelResults.elo_rank)))
            const winRate = (vicunaModelResults.num_wins + vicunaModelResults.num_ties / 2) / vicunaModelResults.num_matches
            createTableScoreCell(rowE, createTextE(round(winRate)))
        } else {
            rowE.insertCell()
            rowE.insertCell()
        }

        if (benchmarks.includes('lm-evaluation-harness'))
            createTableScoreCell(rowE, createTextE(round(averageLmEvaluationHarnessScores[model])))
        else
            createTableScoreCell(rowE, createTextE(''))
    }

    return tableE
}

export async function createBenchmarksV(baseUrl) {
    const hashParameters = parseHash()
    if (hashParameters.has('benchmark'))
        return createSingleBenchmarkV(baseUrl, hashParameters.get('benchmark'), hashParameters)
    return await createBenchmarksIndexV(baseUrl)
}
