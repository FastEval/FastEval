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

    const modelPairs = []
    for (const model1Name of modelNames) {
        for (const model2Name of modelNames) {
            if (model1Name === model2Name)
                continue
            modelPairs.push([model1Name, model2Name])
        }
    }

    const performanceDifferences = new Map()
    for (const modelPair of modelPairs) {
        const [model1Name, model2Name] = modelPair

        const commonBenchmarks = modelsByName[model1Name].benchmarks
            .filter(benchmark => modelsByName[model2Name].benchmarks.includes(benchmark))

        let performanceDifference = 0
        for (const benchmarkName of commonBenchmarks) {
            const model1Performance = getScore(model1Name, commonBenchmarks, benchmarkName)
            const model2Performance = getScore(model2Name, commonBenchmarks, benchmarkName)
            performanceDifference += (model1Performance - model2Performance) / commonBenchmarks.length
        }

        performanceDifferences.set(modelPair, performanceDifference)
    }

    function lossf(rankings) {
        return modelPairs.map(modelPair => {
            const [model1Name, model2Name] = modelPair
            const performanceDifference = performanceDifferences.get(modelPair)
            const rankDifference = rankings.get(model1Name) - rankings.get(model2Name)
            return (rankDifference > 0 && performanceDifference < 0 || rankDifference < 0 && performanceDifference > 0) ? 1 : 0
        }).reduce((a, b) => a + b, 0)
    }

    const populationSize = 100
    let population = []
    for (let i = 0; i < populationSize; i++) {
        const rankings = new Map(modelNames.map(modelName => [modelName, Math.random()]))
        const loss = lossf(rankings)
        population.push([rankings, loss])
    }

    const numIterations = 20_000
    for (let i = 0; i < numIterations; i++) {
        const currentItemIndex = Math.floor(Math.random() * population.length)
        const [currentRanking, currentLoss] = population[currentItemIndex]

        const newRanking = new Map(currentRanking)
        for (const modelName of newRanking.keys())
            newRanking.set(modelName, newRanking.get(modelName) + (Math.random() > 0.5 ? 1 : -1) * (1 - i / numIterations))
        const newLoss = lossf(newRanking)

        if (newLoss > currentLoss)
            continue
        else if (newLoss === currentLoss)
            population[currentItemIndex] = [newRanking, newLoss]
        else
            population.push([newRanking, newLoss])

        if (i % Math.round(populationSize / 5) === 0)
            population = population.toSorted(([ranking1, loss1], [ranking2, loss2]) => loss1 - loss2)
                .slice(0, Math.ceil((1 - i / numIterations) * populationSize))
    }

    return Object.fromEntries(population.toSorted(([ranking1, loss1], [ranking2, loss2]) => loss1 - loss2)[0][0].entries())
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

    const tableE = document.createElement('table')

    const theadE = tableE.createTHead().insertRow()
    theadE.insertCell().appendChild(createTextE('Rank'))
    theadE.insertCell().appendChild(createTextE('Model'))
    theadE.insertCell().appendChild(createTextE('Total'))
    theadE.insertCell().appendChild(createLinkE('lm-evaluation-harness', { benchmark: 'lm-evaluation-harness' }))
    theadE.insertCell().appendChild(createLinkE('Vicuna Elo Rank', { benchmark: 'vicuna' }))
    theadE.insertCell().appendChild(createLinkE('OpenAI Evals', { benchmark: 'openai-evals' }))
    theadE.insertCell().appendChild(createTextE('HumanEval+'))
    const tbodyE = tableE.createTBody()

    for (const [position, { model_name: model, benchmarks }] of modelsSortedByRank.entries()) {
        const rowE = tbodyE.insertRow()

        createTableScoreCell(rowE, createTextE(position + 1))

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
