import { createTextE } from '../components/text.js'
import { createBackToMainPageE } from '../components/back-to-main-page.js'
import { createEvaluationsMap, fetchEvaluations, fetchFiles, round } from '../utils.js'
import { createModelLinkE } from '../components/model-link.js'
import { createTableScoreCell } from '../components/table-score-cell.js'

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

    const evaluations = await fetchEvaluations(baseUrl)
    const evaluationsMap = createEvaluationsMap(evaluations)
    const scores = await fetchFiles(baseUrl, evaluations, 'lm-evaluation-harness', 'total.json')

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
        return scores.get(id).tasks[taskId]
    }

    const ids = Array.from(scores.keys())
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

    const idsSortedByAverageScore = Array.from(scores.entries())
        .toSorted(([id1, scores1], [id2, scores2]) => scores2.average - scores1.average).map(([id, score]) => id)

    for (const id of idsSortedByAverageScore) {
        const rowE = tableBodyE.insertRow()
        rowE.insertCell().appendChild(createModelLinkE(evaluationsMap.get(id)))

        createTableScoreCell(rowE, createTextE(round(scores.get(id).average)))

        rowE.insertCell()

        for (const taskId of taskIds)
            createTableScoreCell(rowE, createTextE(round(getScore(id, taskId))), getRelativeScore(id, taskId))
    }

    return containerE
}
