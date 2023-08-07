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

async function createSingleBenchmarkV(baseUrl, benchmarkName, parameters) {
    switch (benchmarkName) {
        case 'lm-evaluation-harness':
            return await LMEvaluationHarness.createV(baseUrl)
        case 'human-eval-plus':
            return await HumanEvalPlus.createV(baseUrl, parameters)
        case 'cot':
            return await CoT.createV(baseUrl, parameters)
        case 'mt-bench':
            return await MTBench.createV(baseUrl, parameters)
        default:
            throw new Error()
    }
}

function computeEvaluationRanks(evaluations, getScore, getTotalScore) {
    const ids = evaluations.map(({ id }) => id)
    const idToEvaluationInformation = new Map(evaluations.map(evaluation => [evaluation.id, evaluation]))

    const totalScores = {}
    for (const id of ids)
        totalScores[id] = getTotalScore(id, idToEvaluationInformation.get(id).benchmarks)

    const initialOrderTotalScore = ids.filter(id => totalScores[id] !== null)
        .toSorted((id1, id2) => totalScores[id2] - totalScores[id1])
    const initialOrderBaseModels = ids.filter(id =>
        idToEvaluationInformation.get(id).benchmarks.length === 1 && idToEvaluationInformation.get(id).benchmarks[0] === 'lm-evaluation-harness')
        .toSorted((id1, id2) => getScore(id2, 'lm-evaluation-harness') - getScore(id1, 'lm-evaluation-harness'))
    const initialFixedEvaluations = initialOrderTotalScore.concat(initialOrderBaseModels)
    const initialFixedScores = Object.fromEntries(initialFixedEvaluations.map((id, index) => [id, initialFixedEvaluations.length - index]))
    const remainingEvaluations = ids.filter(id => !initialFixedEvaluations.includes(id))
    const minimumRemainingScore = initialOrderBaseModels.length

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
                newRanking.set(id, 1 + minimumRemainingScore + (Math.random() * (ids.length - minimumRemainingScore)))
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

function createHowToReadThisLeaderboardV() {
    const containerE = document.createElement('details')

    const summaryE = document.createElement('summary')
    summaryE.classList.add('how-to-read-leaderboard__summary')
    summaryE.textContent = 'How to read this leaderboard?'
    containerE.appendChild(summaryE)

    const detailsE = document.createElement('ul')
    detailsE.classList.add('how-to-read-leaderboard__details')
    containerE.appendChild(detailsE)

    const greenEqualBetterE = document.createElement('li')
    greenEqualBetterE.classList.add('how-to-read-leaderboard__color-green')
    greenEqualBetterE.textContent = '___ = Higher score = Better.'
    detailsE.appendChild(greenEqualBetterE)

    const redEqualWorseE = document.createElement('li')
    redEqualWorseE.classList.add('how-to-read-leaderboard__color-red')
    redEqualWorseE.textContent = '___ = Lower score = Worse.'
    detailsE.appendChild(redEqualWorseE)

    const sizeE = document.createElement('li')
    sizeE.classList.add('how-to-read-leaderboard__space')
    sizeE.textContent = 'Size: Size of the model in billions of parameters. Larger = Requires more resources = Worse. But usually gives better scores.'
    detailsE.appendChild(sizeE)

    const mtBenchE = document.createElement('li')
    mtBenchE.classList.add('how-to-read-leaderboard__space')
    mtBenchE.textContent = 'MT-Bench: Measures conversational capabilities.'
    detailsE.appendChild(mtBenchE)

    const cotE = document.createElement('li')
    cotE.textContent = 'CoT (Chain-Of-Thought): Measures multi-step reasoning capabilities.'
    detailsE.appendChild(cotE)

    const humanEvalE = document.createElement('li')
    humanEvalE.textContent = "HumanEval+: Measures Python coding performance. Humans were only involved in the creation of the evaluation dataset, the name is misleading."
    detailsE.appendChild(humanEvalE)

    const lmEvalE = document.createElement('li')
    lmEvalE.textContent = "LM-Eval: Measures general capabilities, but doesn't use the model-specific prompt templates."
    detailsE.appendChild(lmEvalE)

    const clickOnColumnsE = document.createElement('li')
    clickOnColumnsE.classList.add('how-to-read-leaderboard__space')
    clickOnColumnsE.textContent = 'Click on the columns to see the benchmark-specific leaderboards with more details.'
    detailsE.appendChild(clickOnColumnsE)

    return containerE
}

export async function fetchAndProcessReports(baseUrl, models) {
    const [
        lmEvaluationHarnessResults,
        humanEvalPlusResults,
        cotResults,
        mtBenchResults,
    ] = await Promise.all([
        fetchFiles(baseUrl, models, 'lm-evaluation-harness', 'gpt4all.json'),
        fetchFiles(baseUrl, models, 'human-eval-plus', '/scores.json'),
        fetchFiles(baseUrl, models, 'cot', '/scores.json'),
        fetchFiles(baseUrl, models, 'mt-bench', '/scores.json'),
    ])

    for (const report of lmEvaluationHarnessResults.values())
        report.absoluteScore = LMEvaluationHarness.computeAverageScore(report.results)

    for (const report of humanEvalPlusResults.values())
        report.absoluteScore = report.scores.plus

    const modelById = new Map(models.map(modelInformation => [modelInformation.id, modelInformation]))

    function getScore(id, benchmarkName) {
        const benchmarks = modelById.get(id).benchmarks
        if (!benchmarks.includes(benchmarkName))
            return null

        if (benchmarkName === 'lm-evaluation-harness')
            return lmEvaluationHarnessResults.get(id).absoluteScore
        else if (benchmarkName === 'human-eval-plus')
            return humanEvalPlusResults.get(id).absoluteScore
        else if (benchmarkName === 'cot')
            return cotResults.get(id).total
        else if (benchmarkName === 'mt-bench')
            return mtBenchResults.get(id).average

        return null
    }

    return getScore
}

export async function createBenchmarksIndexV(baseUrl) {
    const containerE = document.createElement('div')

    containerE.appendChild(createHowToReadThisLeaderboardV())

    const explanationE = document.createElement('div')
    explanationE.classList.add('main__explanation')
    const informationLinkE = document.createElement('a')
    informationLinkE.textContent = 'GitHub repository'
    informationLinkE.href = 'https://github.com/FastEval/FastEval'
    explanationE.append(
        createTextE('See the '),
        informationLinkE,
        createTextE(' for more information.')
    )
    containerE.appendChild(explanationE)

    const evaluations = await fetchEvaluations(baseUrl)
    const getScore = await fetchAndProcessReports(baseUrl, evaluations)

    const allBenchmarks = ['mt-bench', 'cot', 'human-eval-plus', 'lm-evaluation-harness']

    const benchmarkMinimums = new Map()
    const benchmarkMaximums = new Map()
    for (const benchmarkName of allBenchmarks) {
        for (const { id } of evaluations) {
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
        if (!benchmarks.includes('lm-evaluation-harness'))
            return null
        if (!benchmarks.includes('human-eval-plus'))
            return null
        if (!benchmarks.includes('cot'))
            return null
        if (!benchmarks.includes('mt-bench'))
            return null

        // https://github.com/FastEval/FastEval/issues/61#issuecomment-1668562791
        const totalScore =
            2.258328740981252 * getScore(id, 'mt-bench')
            + 15.877679229809127 * getScore(id, 'cot')
            + 15.128786199627087 * getScore(id, 'human-eval-plus')
            + 0.4641024716075128 * getScore(id, 'lm-evaluation-harness')
        return totalScore
    }

    const evaluationRanks = computeEvaluationRanks(evaluations, getRelativeScore, getTotalScore)
    const evaluationsSortedByRank = evaluations.toSorted((evaluation1, evaluation2) => {
        const evaluation1Rank = evaluationRanks[evaluation1.id]
        const evaluation2Rank = evaluationRanks[evaluation2.id]
        return evaluation2Rank - evaluation1Rank
    })

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

    const theadE = tableE.createTHead().insertRow()
    theadE.insertCell().appendChild(createTextE('Rank'))
    theadE.insertCell().appendChild(createTextE('Size'))
    theadE.insertCell().appendChild(createTextE('Model'))
    theadE.insertCell().appendChild(createTextE('Total'))
    theadE.insertCell()
    theadE.insertCell().appendChild(createLinkE('MT-Bench', { benchmark: 'mt-bench' }))
    theadE.insertCell().appendChild(createLinkE('CoT', { benchmark: 'cot' }))
    theadE.insertCell().appendChild(createLinkE('HumanEval+', { benchmark: 'human-eval-plus' }))
    theadE.insertCell().appendChild(createLinkE('LM-Eval', { benchmark: 'lm-evaluation-harness' }))
    const tbodyE = tableE.createTBody()

    let didInsertSeparatorToBaseModels = false
    for (const [position, evaluationInformation] of evaluationsSortedByRank.entries()) {
        const { id, model_name: modelName, benchmarks } = evaluationInformation

        let rowE = tbodyE.insertRow()

        if (benchmarks.length === 1 && benchmarks[0] === 'lm-evaluation-harness') {
            if (!didInsertSeparatorToBaseModels) {
                const separatorRowE = rowE.insertCell()
                separatorRowE.setAttribute('colspan', (5 + allBenchmarks.length).toString())
                separatorRowE.classList.add('separator-row')
                separatorRowE.textContent = 'Base models. Not evaluated on instruction-model specific benchmarks.'
                rowE = tbodyE.insertRow()
                didInsertSeparatorToBaseModels = true
            }

            createTableScoreCell(rowE, createTextE('(' + (position + 1) + ')'))
        } else {
            createTableScoreCell(rowE, createTextE(position + 1))
        }

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

export async function createBenchmarksV(baseUrl) {
    const hashParameters = parseHash()
    if (hashParameters.has('benchmark'))
        return createSingleBenchmarkV(baseUrl, hashParameters.get('benchmark'), hashParameters)
    return await createBenchmarksIndexV(baseUrl)
}
