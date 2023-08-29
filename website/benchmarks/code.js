import { fetchEvaluations, fetchFiles, round } from '../utils.js'
import { createTextE } from '../components/text.js'
import { createModelLinkE } from '../components/model-link.js'
import { createTableScoreCell } from '../components/table-score-cell.js'

export async function createE(baseUrl) {
    const containerE = document.createElement('div')

    const evaluations = await fetchEvaluations(baseUrl)
    const scores = await fetchFiles(baseUrl, evaluations, 'total', 'scores.json')

    const tableE = document.createElement('table')
    containerE.appendChild(tableE)

    const tableHeadE = tableE.createTHead().insertRow()
    const tableBodyE = tableE.createTBody()

    tableHeadE.insertCell().appendChild(createTextE('Model'))
    tableHeadE.insertCell().appendChild(createTextE('Total'))
    tableHeadE.insertCell()

    const tasks = [
        ['human-eval-plus', 'HumanEval+'],
        ['ds1000', 'DS-1000-Chat'],
    ]

    const ids = Array.from(scores.keys())
    const taskIdsWithScores = tasks.map(([taskId, _]) => [taskId, ids.map(id => scores.get(id).benchmarks[taskId])])
    const taskIdToMinimumScore = Object.fromEntries(taskIdsWithScores.map(([taskId, scores]) => [taskId, Math.min(...scores)]))
    const taskIdToMaximumScore = Object.fromEntries(taskIdsWithScores.map(([taskId, scores]) => [taskId, Math.max(...scores)]))

    const totals = ids.map(id => scores.get(id).code)
    const minTotal = Math.min(...totals)
    const maxTotal = Math.max(...totals)

    function getRelativeScore(id, taskId) {
        return (scores.get(id).benchmarks[taskId] - taskIdToMinimumScore[taskId]) / (taskIdToMaximumScore[taskId] - taskIdToMinimumScore[taskId])
    }

    for (const [taskId, taskName] of tasks)
        tableHeadE.insertCell().appendChild(createTextE(taskName))

    const evaluationsSortedByAverageScore = Array.from(scores.entries())
        .toSorted(([id1, scores1], [id2, scores2]) => scores2.code - scores1.code)

    for (const [id, scores] of evaluationsSortedByAverageScore) {
        const rowE = tableBodyE.insertRow()
        rowE.insertCell().appendChild(createModelLinkE(evaluations.get(id)))

        createTableScoreCell(rowE, createTextE(round(scores.code)), (scores.code - minTotal) / (maxTotal - minTotal))

        rowE.insertCell()

        for (const [taskId, taskName] of tasks)
            createTableScoreCell(rowE, createTextE(round(scores.benchmarks[taskId])), getRelativeScore(id, taskId))
    }

    console.log(scores)

    return containerE
}
