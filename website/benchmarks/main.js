import { round, parseHash, fetchEvaluations, fetchFiles } from '../utils.js'
import { createLinkE } from '../components/link.js'
import { createTextE } from '../components/text.js'
import { createTableScoreCell } from '../components/table-score-cell.js'
import * as LMEvaluationHarness from '../benchmarks/lm-evaluation-harness.js'
import * as HumanEvalPlus from '../benchmarks/human-eval-plus.js'
import * as CoT from '../benchmarks/cot.js'
import * as MTBench from '../benchmarks/mt-bench.js'
import { createModelLinkE } from '../components/model-link.js'
import { getModelNumParams } from '../utils.js'

async function createSingleBenchmarkE(baseUrl, benchmarkName, parameters) {
    switch (benchmarkName) {
        case 'lm-evaluation-harness':
            return await LMEvaluationHarness.createE(baseUrl)
        case 'human-eval-plus':
            return await HumanEvalPlus.createE(baseUrl, parameters)
        case 'cot':
            return await CoT.createE(baseUrl, parameters)
        case 'mt-bench':
            return await MTBench.createE(baseUrl, parameters)
        default:
            throw new Error()
    }
}

function computeEvaluationRanks(evaluations, getScore, getTotalScore) {
    const ids = Array.from(evaluations.keys())
    const idToEvaluationInformation = evaluations

    const totalScores = {}
    for (const id of ids)
        totalScores[id] = getTotalScore(id, idToEvaluationInformation.get(id).benchmarks)

    const initialFixedEvaluations = ids.filter(id => totalScores[id] !== null)
        .toSorted((id1, id2) => totalScores[id2] - totalScores[id1])
    const initialFixedScores = Object.fromEntries(initialFixedEvaluations.map((id, index) => [id, initialFixedEvaluations.length - index]))
    const remainingEvaluations = ids.filter(id => !initialFixedEvaluations.includes(id))

    const evaluationPairs = []
    for (const [i, id1] of ids.entries()) {
        for (const [j, id2] of ids.entries()) {
            if (i === j)
                break
            evaluationPairs.push([id1, id2])
        }
    }

    const performanceDifferences = new Map()
    for (const evaluationPair of evaluationPairs) {
        const [id1, id2] = evaluationPair

        const commonBenchmarks = idToEvaluationInformation.get(id1).benchmarks
            .filter(benchmark => idToEvaluationInformation.get(id2).benchmarks.includes(benchmark))

        if (commonBenchmarks.length === 1 && commonBenchmarks[0] === 'lm-evaluation-harness') {
            const evaluation1NumBenchmarks = idToEvaluationInformation.get(id1).benchmarks.length
            const evaluation2NumBenchmarks = idToEvaluationInformation.get(id2).benchmarks.length
            if (evaluation1NumBenchmarks === 1 && evaluation2NumBenchmarks !== 1) {
                performanceDifferences.set(evaluationPair, -Infinity)
                continue
            } else if (evaluation1NumBenchmarks !== 1 && evaluation2NumBenchmarks === 1) {
                performanceDifferences.set(evaluationPair, Infinity)
                continue
            }
        }

        let performanceDifference = 0
        for (const benchmarkName of commonBenchmarks) {
            const evaluation1Performance = getScore(id1, benchmarkName)
            const evaluation2Performance = getScore(id2, benchmarkName)
            performanceDifference += (evaluation1Performance - evaluation2Performance) / commonBenchmarks.length
        }

        if (performanceDifference !== 0)
            performanceDifferences.set(evaluationPair, performanceDifference)
    }

    function lossf(rankings) {
        return evaluationPairs.map(evaluationPair => {
            const [id1, id2] = evaluationPair
            if (!performanceDifferences.has(evaluationPair))
                return 0
            const performanceDifference = performanceDifferences.get(evaluationPair)
            const rankDifference = rankings.get(id1) - rankings.get(id2)
            if (rankDifference > 0 && performanceDifference === -Infinity)
                return 1e6 * rankDifference
            if (rankDifference < 0 && performanceDifference === Infinity)
                return 1e6 * (-rankDifference)
            return (rankDifference > 0 && performanceDifference < 0 || rankDifference < 0 && performanceDifference > 0) ? 1 : 0
        }).reduce((a, b) => a + b, 0)
    }

    function renormalize(rankings) {
        return new Map([...rankings.entries()]
            .toSorted(([id1, evaluation1Rank], [id2, evaluation2Rank]) => evaluation2Rank - evaluation1Rank)
            .map(([id, previousEvaluationRank], index) => [id, ids.length - index]))
    }

    const initialPopulationSize = 100
    const minPopulationSize = 20
    let population = []
    for (let i = 0; i < initialPopulationSize; i++) {
        const rankings = renormalize(new Map(ids.map(id =>
            [id, initialFixedScores[id] ?? (Math.random()) * initialFixedEvaluations.length])))
        const loss = lossf(rankings)
        population.push([rankings, loss])
    }

    const numIterations = 10_000
    for (let i = 0; i < numIterations; i++) {
        const currentItemIndex = Math.floor(Math.random() * population.length)
        const [currentRanking, currentLoss] = population[currentItemIndex]

        let newRanking = new Map(currentRanking)

        for (const id of remainingEvaluations) {
            if (Math.random() < 1 / remainingEvaluations.length)
                newRanking.set(id, 1 + (Math.random() * ids.length))
        }

        newRanking = renormalize(newRanking)

        const newLoss = lossf(newRanking)
        if (newLoss <= currentLoss)
            population.push([newRanking, newLoss])

        if (i % Math.round(initialPopulationSize / 5) === 0)
            population = population.toSorted(([ranking1, loss1], [ranking2, loss2]) => loss1 - loss2)
                .slice(0, Math.max(minPopulationSize, Math.ceil((1 - i / numIterations) * initialPopulationSize)))
    }

    const populationSortedByLoss = population.toSorted(([ranking1, loss1], [ranking2, loss2]) => loss1 - loss2)
    const lowestLoss = populationSortedByLoss[0][1]
    const populationItemsWithLowestLoss = populationSortedByLoss.filter(([ranking, loss]) => loss === lowestLoss)
        .map(([ranking, loss]) => [...ranking.entries()]
            .toSorted(([id1, evaluation1Rank], [id2, evaluation2Rank]) => evaluation2Rank - evaluation1Rank))

    let orderings = populationItemsWithLowestLoss
    for (let i = ids.length - 1; i >= 0; i--)
        orderings = orderings.toSorted((ordering1, ordering2) => ordering1[i][0].localeCompare(ordering2[i][0]))

    return Object.fromEntries(orderings[0])
}

export async function createBenchmarksIndexE(baseUrl) {
    const containerE = document.createElement('div')

    const evaluations = await fetchEvaluations(baseUrl)
    const scores = await fetchFiles(baseUrl, evaluations, 'total', 'scores.json')

    function getScore(id, benchmarkName) {
        return scores.get(id).benchmarks[benchmarkName] || null
    }

    const allBenchmarks = ['mt-bench', 'cot', 'human-eval-plus', 'lm-evaluation-harness']

    const benchmarkMinimums = new Map()
    const benchmarkMaximums = new Map()
    for (const benchmarkName of allBenchmarks) {
        for (const id of evaluations.keys()) {
            const score = getScore(id, benchmarkName)
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

    function getRelativeScore(id, benchmarkName) {
        const score = getScore(id, benchmarkName)
        return (score - benchmarkMinimums.get(benchmarkName))
            / (benchmarkMaximums.get(benchmarkName) - benchmarkMinimums.get(benchmarkName))
    }

    function getTotalScore(id, benchmarks) {
        return scores.get(id).total || null
    }

    const evaluationRanks = computeEvaluationRanks(evaluations, getRelativeScore, getTotalScore)
    const evaluationsSortedByRank = Array.from(evaluations.entries()).toSorted(([id1, evaluation1], [id2, evaluation2]) =>
        evaluationRanks[id2] - evaluationRanks[id1]).map(e => e[1])

    const allNumParameters = []
    for (const evaluationInformation of evaluationsSortedByRank) {
        const numParameters = getModelNumParams(evaluationInformation)
        if (numParameters === '' || numParameters === 'proprietary')
            continue
        allNumParameters.push(parseInt(numParameters.replace('B', '')))
    }

    const minNumParametersLog = Math.log2(Math.min(...allNumParameters))
    const maxNumParametersLog = Math.log2(Math.max(...allNumParameters))

    let allTotalScores = []
    for (const evaluationInformation of evaluationsSortedByRank) {
        const { id, benchmarks } = evaluationInformation
        const totalScore = getTotalScore(id, benchmarks)
        if (totalScore !== null)
            allTotalScores.push(totalScore)
    }

    const minTotalScore = Math.min(...allTotalScores)
    const maxTotalScore = Math.max(...allTotalScores)

    const tableE = document.createElement('table')
    tableE.classList.add('main__table')
    containerE.appendChild(tableE)

    const githubLinkE = document.createElement('a')
    githubLinkE.href = 'https://github.com/FastEval/FastEval'
    const githubLinkImgE = document.createElement('img')
    githubLinkImgE.src = './website/img/github-mark.svg'
    githubLinkImgE.classList.add('main-page-github-icon')
    githubLinkE.prepend(githubLinkImgE)

    const theadE = tableE.createTHead()
    const theadClickHereE = theadE.insertRow()
    theadClickHereE.classList.add('no-td-border')
    const githubLinkCellE = theadClickHereE.insertCell()
    githubLinkCellE.colSpan = 5
    githubLinkCellE.append(githubLinkE)
    const theadClickHereTextE = theadClickHereE.insertCell()
    theadClickHereTextE.colSpan = 4
    theadClickHereTextE.style['text-align'] = 'center'
    theadClickHereTextE.appendChild(createTextE('⤹ Click on the links for more details! ⤸'))
    theadClickHereTextE.classList.add('nowrap')
    const theadRowE = theadE.insertRow()
    theadRowE.insertCell().appendChild(createTextE('Rank'))
    theadRowE.insertCell().appendChild(createTextE('Size'))
    theadRowE.insertCell().appendChild(createTextE('Model'))
    theadRowE.insertCell().appendChild(createTextE('Total'))
    theadRowE.insertCell()
    const mtBenchHeaderE = createLinkE('MT-Bench', { benchmark: 'mt-bench' })
    mtBenchHeaderE.classList.add('nowrap')
    theadRowE.insertCell().appendChild(mtBenchHeaderE)
    theadRowE.insertCell().appendChild(createLinkE('CoT', { benchmark: 'cot' }))
    theadRowE.insertCell().appendChild(createLinkE('HumanEval+', { benchmark: 'human-eval-plus' }))
    const lmEvalHeaderE = createLinkE('LM-Eval', { benchmark: 'lm-evaluation-harness' })
    lmEvalHeaderE.classList.add('nowrap')
    theadRowE.insertCell().appendChild(lmEvalHeaderE)
    const tbodyE = tableE.createTBody()

    for (const [position, evaluationInformation] of evaluationsSortedByRank.entries()) {
        const { id, model_name: modelName, benchmarks } = evaluationInformation

        let rowE = tbodyE.insertRow()

        createTableScoreCell(rowE, createTextE(position + 1))

        const numParameters = getModelNumParams(evaluationInformation)
        if (numParameters === '') {
            createTableScoreCell(rowE, createTextE(numParameters))
        } else if (numParameters === 'proprietary') {
            createTableScoreCell(rowE, createTextE(''), -1)
        } else {
            const color = 1 - ((Math.log2(parseInt(numParameters.replace('B', ''))) - minNumParametersLog)
                / (maxNumParametersLog - minNumParametersLog))
            createTableScoreCell(rowE, createTextE(numParameters), color)
        }

        rowE.insertCell().appendChild(createModelLinkE(evaluationInformation))

        const totalScore = getTotalScore(id, benchmarks)
        if (totalScore === null)
            createTableScoreCell(rowE, createTextE(''))
        else
            createTableScoreCell(rowE, createTextE(round(totalScore)), (totalScore - minTotalScore) / (maxTotalScore - minTotalScore))

        rowE.insertCell()

        for (const benchmarkName of allBenchmarks) {
            const score = getScore(id, benchmarkName)
            if (score === null) {
                createTableScoreCell(rowE, createTextE(''))
                continue
            }

            const relativeScore = getRelativeScore(id, benchmarkName)
            createTableScoreCell(rowE, createTextE(round(score)), relativeScore)
        }
    }

    return containerE
}

export async function createBenchmarksE(baseUrl) {
    const hashParameters = parseHash()
    if (hashParameters.has('benchmark'))
        return createSingleBenchmarkE(baseUrl, hashParameters.get('benchmark'), hashParameters)
    return await createBenchmarksIndexE(baseUrl)
}
