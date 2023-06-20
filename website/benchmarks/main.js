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

function computeModelRanksSingleSeed(modelNames, allBenchmarks, modelsByName, getScore) {
    const SCALE = 400
    const BASE = 10
    const K = 32

    const modelRanks = Object.fromEntries(modelNames.map(modelName => [modelName, 1000]))
    for (let i = 0; i < 100; i++) {
        const model1Name = modelNames[Math.floor(Math.random() * modelNames.length)]
        const model2Name = modelNames[Math.floor(Math.random() * modelNames.length)]

        let model1Score
        let model2Score
        for (let j = 0; j < 1; j++) {
            const benchmarkName = allBenchmarks[Math.floor(Math.random() * allBenchmarks.length)]

            const model1 = modelsByName[model1Name]
            const model2 = modelsByName[model2Name]

            model1Score = getScore(model1Name, model1.benchmarks, benchmarkName)
            model2Score = getScore(model2Name, model2.benchmarks, benchmarkName)

            if (model1Score !== null && model2Score !== null)
                break
        }

        if (model1Score === null || model2Score === null)
            continue

        const e1 = 1 / (1 + BASE ** ((modelRanks[model2Name] - modelRanks[model1Name]) / SCALE))
        const e2 = 1 / (1 + BASE ** ((modelRanks[model1Name] - modelRanks[model2Name]) / SCALE))

        const sa = model1Score > model2Score ? 1
            : model2Score > model1Score ? 0
            : 0.5

        modelRanks[model1Name] += K * (sa - e1)
        modelRanks[model2Name] += K * (1 - sa - e2)
    }

    return modelRanks
}

function computeModelRanks(models, getScore, allBenchmarks) {
    const modelNames = [...new Set(models.map(({ model_name: model }) => model))]
    const modelsByName = Object.fromEntries(models.map(model => [model.model_name, model]))

    const numSeeds = 1000
    const modelRanks = Object.fromEntries(modelNames.map(modelName => [modelName, 0]))
    for (let i = 0; i < numSeeds; i++) {
        const ranks = computeModelRanksSingleSeed(modelNames, allBenchmarks, modelsByName, getScore)
        for (const [modelName, rank] of Object.entries(ranks)) {
            modelRanks[modelName] += rank / numSeeds
        }
    }

    return modelRanks
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

    const allBenchmarks = ['lm-evaluation-harness', 'vicuna', 'openai-evals', 'human-eval-plus']

    const benchmarkMinimums = new Map()
    const benchmarkMaximums = new Map()
    for (const benchmarkName of allBenchmarks) {
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

    function getRelativeScore(model, benchmarks, benchmarkName) {
        const score = getScore(model, benchmarks, benchmarkName)
        return (score - benchmarkMinimums.get(benchmarkName))
            / (benchmarkMaximums.get(benchmarkName) - benchmarkMinimums.get(benchmarkName))
    }

    const modelRanks = computeModelRanks(models, getRelativeScore, allBenchmarks)
    const modelsSortedByRank = models.toSorted((model1, model2) => {
        const model1Rank = modelRanks[model1.model_name]
        const model2Rank = modelRanks[model2.model_name]
        return model2Rank - model1Rank
    })

    const modelRanksMin = Math.min(...Object.entries(modelRanks).map(([modelName, rank]) => rank))
    const modelRanksMax = Math.max(...Object.entries(modelRanks).map(([modelName, rank]) => rank))
    const normalizedModelRanks = Object.fromEntries(Object.entries(modelRanks).map(([modelName, modelRank]) =>
        [modelName, (modelRank - modelRanksMin) / (modelRanksMax - modelRanksMin)]))

    const tableE = document.createElement('table')

    const theadE = tableE.createTHead().insertRow()
    theadE.insertCell().appendChild(createTextE('Total Rank'))
    theadE.insertCell().appendChild(createTextE('Model'))
    theadE.insertCell().appendChild(createTextE('Total'))
    theadE.insertCell().appendChild(createLinkE('lm-evaluation-harness', { benchmark: 'lm-evaluation-harness' }))
    theadE.insertCell().appendChild(createLinkE('Vicuna Elo Rank', { benchmark: 'vicuna' }))
    theadE.insertCell().appendChild(createLinkE('OpenAI Evals', { benchmark: 'openai-evals' }))
    theadE.insertCell().appendChild(createTextE('HumanEval+'))
    const tbodyE = tableE.createTBody()

    for (const { model_name: model, benchmarks } of modelsSortedByRank) {
        const rowE = tbodyE.insertRow()

        createTableScoreCell(rowE, createTextE(round(normalizedModelRanks[model])))

        rowE.insertCell().appendChild(createTextE(model))

        const allBenchmarkEvaluated = benchmarks.includes('lm-evaluation-harness')
            && benchmarks.includes('vicuna') && model in vicunaEvaluationResults.models
            && benchmarks.includes('openai-evals')
            && benchmarks.includes('human-eval-plus')
        if (allBenchmarkEvaluated) {
            let relativeAverageScore = 0
            for (const benchmarkName of allBenchmarks)
                relativeAverageScore += getRelativeScore(model, benchmarks, benchmarkName) / allBenchmarks.length

            createTableScoreCell(rowE, createTextE(round(relativeAverageScore)))
        } else {
            createTableScoreCell(rowE, createTextE(''))
        }

        for (const benchmarkName of allBenchmarks) {
            const score = getScore(model, benchmarks, benchmarkName)
            if (score === null) {
                createTableScoreCell(rowE, createTextE(''))
                continue
            }

            let text = round(score)
            if (benchmarkName === 'vicuna')
                text = Math.round(score)
            if (benchmarkName === 'human-eval-plus') {
                createTableScoreCell(rowE, createLinkE(text, { benchmark: 'human-eval-plus', model }))
                continue
            }

            createTableScoreCell(rowE, createTextE(text))
        }
    }

    return tableE
}

export async function createBenchmarksV(baseUrl) {
    const hashParameters = parseHash()
    if (hashParameters.has('benchmark'))
        return createSingleBenchmarkV(baseUrl, hashParameters.get('benchmark'), hashParameters)
    return await createBenchmarksIndexV(baseUrl)
}
