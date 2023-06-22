import { round, parseHash, fetchModels, fetchFiles } from '../utils.js'
import { createLinkE } from '../components/link.js'
import { createTextE } from '../components/text.js'
import { createTableScoreCell } from '../components/table-score-cell.js'
import * as OpenAIEvals from '../benchmarks/openai-evals.js'
import * as Vicuna from '../benchmarks/vicuna.js'
import * as LMEvaluationHarness from '../benchmarks/lm-evaluation-harness.js'
import * as HumanEvalPlus from '../benchmarks/human-eval-plus.js'
import * as CoT from '../benchmarks/cot.js'

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
        case 'cot':
            return await CoT.createV(baseUrl, parameters)
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

    const initialPopulationSize = 100
    const minPopulationSize = 20
    let population = []
    for (let i = 0; i < initialPopulationSize; i++) {
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

        if (i % Math.round(initialPopulationSize / 5) === 0)
            population = population.toSorted(([ranking1, loss1], [ranking2, loss2]) => loss1 - loss2)
                .slice(0, Math.max(minPopulationSize, Math.ceil((1 - i / numIterations) * initialPopulationSize)))
    }

    const populationSortedByLoss = population.toSorted(([ranking1, loss1], [ranking2, loss2]) => loss1 - loss2)
    const lowestLoss = populationSortedByLoss[0][1]
    const populationItemsWithLowestLoss = populationSortedByLoss.filter(([ranking, loss]) => loss === lowestLoss)
        .map(([ranking, loss]) => [...ranking.entries()]
            .toSorted(([model1Name, model1Rank], [model2Name, model2Rank]) => model2Rank - model1Rank))

    let orderings = populationItemsWithLowestLoss
    for (let i = models.length - 1; i >= 0; i--)
        orderings = orderings.toSorted((ordering1, ordering2) => ordering1[i][0].localeCompare(ordering2[i][0]))

    return Object.fromEntries(orderings[0])
}

export async function createBenchmarksIndexV(baseUrl) {
    const containerE = document.createElement('div')

    const explanationE = document.createElement('div')
    explanationE.classList.add('main__explanation')
    const informationLinkE = document.createElement('a')
    informationLinkE.textContent = 'GitHub repository'
    informationLinkE.href = 'https://github.com/tju01/ilm-eval'
    explanationE.append(
        createTextE('See the '),
        informationLinkE,
        createTextE(' for more information.')
    )
    containerE.appendChild(explanationE)

    const models = await fetchModels(baseUrl)

    const [
        vicunaEvaluationResults,
        openaiEvalsResults,
        lmEvaluationHarnessResults,
        humanEvalPlusResults,
        cotResults,
    ] = await Promise.all([
        fetch(baseUrl + '/vicuna/elo.json').then(r => r.json()),
        fetchFiles(baseUrl, models, 'openai-evals', '/__index__.json'),
        fetchFiles(baseUrl, models, 'lm-evaluation-harness'),
        fetchFiles(baseUrl, models, 'human-eval-plus'),
        fetchFiles(baseUrl, models, 'cot', '/scores.json'),
    ])

    const relativeOpenAiEvalsScores = OpenAIEvals.computeRelativeOpenAiEvalsScores(Object.fromEntries(openaiEvalsResults)).averageRelativeScoresByModelName

    const averageLmEvaluationHarnessScores =  Object.fromEntries(lmEvaluationHarnessResults.map(([modelName, results]) =>
        [modelName, LMEvaluationHarness.computeAverageScore(results.results)]))

    const humanEvalPlusResultsMap = Object.fromEntries(humanEvalPlusResults)
    const cotResultsMap = Object.fromEntries(cotResults)

    function getScore(model, benchmarks, benchmarkName) {
        if (!benchmarks.includes(benchmarkName))
            return null

        if (benchmarkName === 'lm-evaluation-harness')
            return averageLmEvaluationHarnessScores[model]
        else if (benchmarkName === 'vicuna' && model in vicunaEvaluationResults)
            return vicunaEvaluationResults[model]
        else if (benchmarkName === 'openai-evals')
            return relativeOpenAiEvalsScores[model]
        else if (benchmarkName === 'human-eval-plus')
            return humanEvalPlusResultsMap[model].score
        else if (benchmarkName === 'cot')
            return cotResultsMap[model].average

        return null
    }

    const allBenchmarks = ['openai-evals', 'vicuna', 'human-eval-plus', 'cot', 'lm-evaluation-harness']

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
    containerE.appendChild(tableE)

    const theadE = tableE.createTHead().insertRow()
    theadE.insertCell().appendChild(createTextE('Rank'))
    theadE.insertCell().appendChild(createTextE('Model'))
    theadE.insertCell().appendChild(createTextE('Total'))
    theadE.insertCell().appendChild(createLinkE('OpenAI Evals', { benchmark: 'openai-evals' }))
    theadE.insertCell().appendChild(createLinkE('Vicuna Elo Rank', { benchmark: 'vicuna' }))
    theadE.insertCell().appendChild(createTextE('HumanEval+'))
    theadE.insertCell().appendChild(createLinkE('CoT', { benchmark: 'cot' }))
    theadE.insertCell().appendChild(createLinkE('lm-evaluation-harness', { benchmark: 'lm-evaluation-harness' }))
    const tbodyE = tableE.createTBody()

    for (const [position, { model_name: model, benchmarks }] of modelsSortedByRank.entries()) {
        const rowE = tbodyE.insertRow()

        createTableScoreCell(rowE, createTextE(position + 1))

        rowE.insertCell().appendChild(createTextE(model))

        const allBenchmarkEvaluated = benchmarks.includes('lm-evaluation-harness')
            && benchmarks.includes('vicuna') && model in vicunaEvaluationResults
            && benchmarks.includes('openai-evals')
            && benchmarks.includes('human-eval-plus')
            && benchmarks.includes('cot')
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

    return containerE
}

export async function createBenchmarksV(baseUrl) {
    const hashParameters = parseHash()
    if (hashParameters.has('benchmark'))
        return createSingleBenchmarkV(baseUrl, hashParameters.get('benchmark'), hashParameters)
    return await createBenchmarksIndexV(baseUrl)
}
