import { createTextE } from '../components/text.js'
import { createBackToMainPageE } from '../components/back-to-main-page.js'
import { createEvaluationsMap, round } from '../utils.js'
import { createModelLinkE } from '../components/model-link.js'
import { createTableScoreCell } from '../components/table-score-cell.js'

export function computeAverageScore(results) {
    return Object.values(results).map(r => r.acc_norm ?? r.acc).reduce((a, b) => a + b, 0) / Object.values(results).length * 100
}

export async function createV(baseUrl) {
    const containerE = document.createElement('div')

    containerE.appendChild(createBackToMainPageE())

    const otherResultsE = document.createElement('div')
    otherResultsE.classList.add('lm-evaluation-harness__other-results')
    containerE.appendChild(otherResultsE)

    const lmEvaluationHarnessLinkE = document.createElement('a')
    lmEvaluationHarnessLinkE.textContent = 'lm-evaluation-harness'
    lmEvaluationHarnessLinkE.href = 'https://github.com/EleutherAI/lm-evaluation-harness'

    const otherResultsLinkE = document.createElement('a')
    otherResultsLinkE.textContent = 'here'
    otherResultsLinkE.href = 'https://gpt4all.io/index.html'

    otherResultsE.append(
        createTextE('This benchmark computes the average over some tasks from '),
        lmEvaluationHarnessLinkE,
        createTextE('. The method is the same as for the gpt4all leaderboard and therefore the numbers are comparable. '
            + 'You can view the gpt4all leaderboard '),
        otherResultsLinkE,
        createTextE('. Scroll down to the section "Performance Benchmarks".'),
    )

    const evaluations = (await (await fetch(baseUrl + '/__index__.json')).json())
        .filter(evaluation => evaluation.benchmarks.includes('lm-evaluation-harness'))
    const evaluationsMap = createEvaluationsMap(evaluations)
    const ids = evaluations.map(evaluation => evaluation.id)

    const results = await Promise.all(ids.map(async id =>
        [id, await fetch(baseUrl + '/lm-evaluation-harness/' + evaluationsMap.get(id).model_name.replace('/', '--') + '/' + id + '/' + 'gpt4all.json').then(r => r.json())]))
    const resultsMap = Object.fromEntries(results)

    const tasks = [
        ['boolq', 'BoolQ'],
        ['piqa', 'PIQA'],
        ['hellaswag', 'HellaSwag'],
        ['winogrande', 'WinoGrande'],
        ['arc_easy', 'ARC-e'],
        ['arc_challenge', 'ARC-c'],
        ['openbookqa', 'OBQA'],
    ]

    function getScore(id, taskId) {
        const r = resultsMap[id].results[taskId]
        return (r.acc_norm ?? r.acc) * 100
    }

    const taskIds = tasks.map(([taskId, taskName]) => taskId)
    const taskIdsWithScores = taskIds.map(taskId => [taskId, ids.map(id => getScore(id, taskId))])
    const taskIdToMinimumScore = Object.fromEntries(taskIdsWithScores.map(([taskId, scores]) => [taskId, Math.min(...scores)]))
    const taskIdToMaximumScore = Object.fromEntries(taskIdsWithScores.map(([taskId, scores]) => [taskId, Math.max(...scores)]))

    function getRelativeScore(id, taskId) {
        return (getScore(id, taskId) - taskIdToMinimumScore[taskId]) / (taskIdToMaximumScore[taskId] - taskIdToMinimumScore[taskId])
    }

    const reportsIndexE = document.createElement('table')
    containerE.appendChild(reportsIndexE)
    const tableHeadE = reportsIndexE.createTHead().insertRow()
    const tableBodyE = reportsIndexE.createTBody()
    tableHeadE.insertCell().appendChild(createTextE('Model'))

    tableHeadE.insertCell().appendChild(createTextE('Average'))
    tableHeadE.insertCell()
    for (const [taskId, taskName] of tasks)
        tableHeadE.insertCell().appendChild(createTextE(taskName))

    const averageScores = Object.fromEntries(ids.map(id => [id, computeAverageScore(resultsMap[id].results)]))
    const idsSortedByAverageScore = ids.sort((id1, id2) => averageScores[id2] - averageScores[id1])

    for (const id of idsSortedByAverageScore) {
        const rowE = tableBodyE.insertRow()
        rowE.insertCell().appendChild(createModelLinkE(evaluationsMap.get(id)))

        createTableScoreCell(rowE, createTextE(round(averageScores[id])))

        rowE.insertCell()

        for (const taskId of taskIds)
            createTableScoreCell(rowE, createTextE(round(getScore(id, taskId))), getRelativeScore(id, taskId))
    }

    return containerE
}
