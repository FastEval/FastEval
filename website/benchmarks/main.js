import { round, parseHash } from '../utils.js'
import { createLinkE } from '../components/link.js'
import { createTextE } from '../components/text.js'
import { createTableScoreCell } from '../components/table-score-cell.js'
import * as OpenAIEvals from '../benchmarks/openai-evals.js'
import * as Vicuna from '../benchmarks/vicuna.js'
import * as LMEvaluationHarness from '../benchmarks/lm-evaluation-harness.js'
import * as HumanEvalPlus from '../benchmarks/human-eval-plus.js'

async function createSingleBenchmarkV(baseUrl, benchmarkName, parameters) {
    switch (benchmarkName) {
        case 'openai-evals':
            return await OpenAIEvals.createV(baseUrl, parameters)
        case 'vicuna':
            return await Vicuna.createV(baseUrl, parameters)
        case 'lm-evaluation-harness':
            return await LMEvaluationHarness.createV(baseUrl)
        case 'human-eval-plus':
            return await HumanEvalPlus.createV(baseUrl, parameters)
        default:
            throw new Error()
    }
}

export async function createBenchmarksIndexV(baseUrl) {
    const models = (await (await fetch(baseUrl + '/__index__.json')).json())

    const [vicunaEvaluationResults, openaiEvalsResults, lmEvaluationHarnessResults, humanEvalPlusResults] = await Promise.all([
        fetch(baseUrl + '/vicuna/reviews.json').then(r => r.json()),
        Promise.all(models.filter(model => model.benchmarks.includes('openai-evals')).map(model => model.model_name)
            .map(async model => [model, await fetch(baseUrl + '/openai-evals/' + model.replace('/', '--') + '/__index__.json').then(r => r.json())])),
        Promise.all(models.filter(model => model.benchmarks.includes('lm-evaluation-harness')).map(model => model.model_name)
            .map(async model => [model, await fetch(baseUrl + '/lm-evaluation-harness/' + model.replace('/', '--') + '.json').then(r => r.json())])),
        Promise.all(models.filter(model => model.benchmarks.includes('human-eval-plus')).map(model => model.model_name)
            .map(async model => [model, await fetch(baseUrl + '/human-eval-plus/' + model.replace('/', '--') + '.json').then(r => r.json())]))
    ])

    const relativeOpenAiEvalsScores = OpenAIEvals.computeRelativeOpenAiEvalsScores(Object.fromEntries(openaiEvalsResults)).averageRelativeScoresByModelName

    const averageLmEvaluationHarnessScores =  Object.fromEntries(lmEvaluationHarnessResults.map(([modelName, results]) =>
        [modelName, LMEvaluationHarness.computeAverageScore(results.results)]))

    const humanEvalPlusResultsMap = Object.fromEntries(humanEvalPlusResults)

    function getScore(model, benchmarks, benchmarkName) {
        if (!benchmarks.includes(benchmarkName))
            return null

        if (benchmarkName === 'lm-evaluation-harness')
            return averageLmEvaluationHarnessScores[model]
        else if (benchmarkName === 'vicuna' && model in vicunaEvaluationResults.models)
            return vicunaEvaluationResults.models[model].elo_rank
        else if (benchmarkName === 'openai-evals')
            return relativeOpenAiEvalsScores[model]
        else if (benchmarkName === 'human-eval-plus')
            return humanEvalPlusResultsMap[model].score

        return null
    }

    const benchmarkMinimums = new Map()
    const benchmarkMaximums = new Map()
    for (const benchmarkName of ['lm-evaluation-harness', 'vicuna', 'openai-evals', 'human-eval-plus']) {
        for (const { model_name: model, benchmarks } of models) {
            const score = getScore(model, benchmarks, benchmarkName)
            if (score === null)
                continue

            if (!benchmarkMinimums.has(benchmarkName))
                benchmarkMinimums.set(benchmarkName, score)
            if (!benchmarkMaximums.has(benchmarkName))
                benchmarkMaximums.set(benchmarkName, score)
            if (benchmarkMinimums.get(benchmarkName) > score)
                benchmarkMinimums.set(benchmarkName, score)
            if (benchmarkMaximums.get(benchmarkName) < score)
                benchmarkMaximums.set(benchmarkName, score)
        }
    }

    const tableE = document.createElement('table')
    const theadE = tableE.createTHead().insertRow()
    theadE.insertCell().appendChild(createTextE('Model'))
    theadE.insertCell().appendChild(createTextE('Total'))
    theadE.insertCell().appendChild(createLinkE('lm-evaluation-harness', { benchmark: 'lm-evaluation-harness' }))
    theadE.insertCell().appendChild(createLinkE('Vicuna Elo Rank', { benchmark: 'vicuna' }))
    theadE.insertCell().appendChild(createLinkE('OpenAI Evals', { benchmark: 'openai-evals' }))
    theadE.insertCell().appendChild(createTextE('HumanEval+'))
    const tbodyE = tableE.createTBody()
    for (const { model_name: model, benchmarks } of models) {
        const rowE = tbodyE.insertRow()
        rowE.insertCell().appendChild(createTextE(model))

        const allBenchmarkEvaluated = benchmarks.includes('lm-evaluation-harness')
            && benchmarks.includes('vicuna') && model in vicunaEvaluationResults.models
            && benchmarks.includes('openai-evals')
            && benchmarks.includes('human-eval-plus')
        if (allBenchmarkEvaluated) {
            const allBenchmarks = ['lm-evaluation-harness', 'vicuna', 'openai-evals', 'human-eval-plus']
            let relativeAverageScore = 0
            for (const benchmarkName of allBenchmarks) {
                const score = getScore(model, benchmarks, benchmarkName)
                const relativeScore = (score - benchmarkMinimums.get(benchmarkName))
                    / (benchmarkMaximums.get(benchmarkName) - benchmarkMinimums.get(benchmarkName))
                relativeAverageScore += relativeScore / allBenchmarks.length
            }

            createTableScoreCell(rowE, createTextE(round(relativeAverageScore)))
        } else {
            createTableScoreCell(rowE, createTextE(''))
        }

        if (benchmarks.includes('lm-evaluation-harness'))
            createTableScoreCell(rowE, createTextE(round(averageLmEvaluationHarnessScores[model])))
        else
            createTableScoreCell(rowE, createTextE(''))

        if (benchmarks.includes('vicuna') && model in vicunaEvaluationResults.models)
            createTableScoreCell(rowE, createTextE(Math.round(vicunaEvaluationResults.models[model].elo_rank)))
        else
            rowE.insertCell()

        if (benchmarks.includes('openai-evals'))
            createTableScoreCell(rowE, createTextE(round(relativeOpenAiEvalsScores[model])))
        else
            rowE.insertCell()

        if (benchmarks.includes('human-eval-plus'))
            createTableScoreCell(rowE, createLinkE(round(humanEvalPlusResultsMap[model].score), { benchmark: 'human-eval-plus', model }))
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
