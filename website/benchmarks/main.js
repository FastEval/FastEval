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

function computeModelRanks(models, getScore, allBenchmarks) {
    const modelNames = [...new Set(models.map(({ model_name: model }) => model))]
    const modelsByName = Object.fromEntries(models.map(model => [model.model_name, model]))

    const performanceDifferences = []

    for (const model1Name of modelNames) {
        for (const model2Name of modelNames) {
            if (model1Name === model2Name)
                continue
            const commonBenchmarks = modelsByName[model1Name].benchmarks
                .filter(benchmark => modelsByName[model2Name].benchmarks.includes(benchmark))
            let performanceDifference = 0
            for (const benchmarkName of commonBenchmarks) {
                const model1Performance = getScore(model1Name, commonBenchmarks, benchmarkName)
                const model2Performance = getScore(model2Name, commonBenchmarks, benchmarkName)
                performanceDifference += (model1Performance - model2Performance) / commonBenchmarks.length
            }

            if (performanceDifference !== 0)
                performanceDifferences.push([model1Name, model2Name, performanceDifference])
        }
    }

    function loss(rankings) {
        let totalLoss = 0
        for (const [model1Name, model2Name, performanceDifference] of performanceDifferences) {
            const rank1 = rankings[model1Name]
            const rank2 = rankings[model2Name]
            const rankDifference = rank1 - rank2
            totalLoss += Math.abs(rankDifference - performanceDifference)
        }

        return totalLoss
    }

    const averageRankings = Object.fromEntries(modelNames.map(modelName => [modelName, 0]))
    for (let j = 0; j < 100; j++) {
        const numIterations = 1_000
        let currentLoss = Infinity
        let rankings = Object.fromEntries(modelNames.map(modelName => [modelName, Math.random()]))
        for (let i = 1; i < numIterations; i++) {
            const modelToChange = modelNames[Math.floor(Math.random() * modelNames.length)]

            const rankingsWithIncrease = { ...rankings }
            rankingsWithIncrease[modelToChange] += 1 - i / numIterations
            let lossIfIncreased = loss(rankingsWithIncrease)

            if (lossIfIncreased < currentLoss) {
                currentLoss = lossIfIncreased
                rankings = rankingsWithIncrease
            }

            const rankingsWithDecrease = { ...rankings }
            rankingsWithDecrease[modelToChange] -= 1 - i / numIterations
            let lossIfDecreased = loss(rankingsWithIncrease)

            if (lossIfDecreased < currentLoss) {
                currentLoss = lossIfDecreased
                rankings = rankingsWithDecrease
            }
        }

        for (const [modelName, modelRank] of Object.entries(rankings)) {
            averageRankings[modelName] += modelRank
        }
    }

    return averageRankings
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
