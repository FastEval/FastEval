import { fetchEvaluations, fetchFiles, round } from '../utils.js'
import { createTextE } from '../components/text.js'
import { createModelLinkE } from '../components/model-link.js'
import { createTableScoreCell } from '../components/table-score-cell.js'

export async function createE(baseUrl, parameters) {
    const containerE = document.createElement('div')

    const evaluations = await fetchEvaluations(baseUrl)
    const scores = await fetchFiles(baseUrl, evaluations, 'ds1000', 'scores.json')

    const tableE = document.createElement('table')
    containerE.appendChild(tableE)

    const tableHeadE = tableE.createTHead().insertRow()
    const tableBodyE = tableE.createTBody()

    tableHeadE.insertCell().appendChild(createTextE('Model'))
    tableHeadE.insertCell().appendChild(createTextE('Average'))
    tableHeadE.insertCell()

    const taskIds = [
        'Matplotlib',
        'Numpy',
        'Pytorch',
        'Scipy',
        'Tensorflow',
        'Pandas',
        'Sklearn',
    ]

    const ids = Array.from(scores.keys())
    const taskIdsWithScores = taskIds.map(taskId => [taskId, ids.map(id => scores.get(id).tasks[taskId])])
    const taskIdToMinimumScore = Object.fromEntries(taskIdsWithScores.map(([taskId, scores]) => [taskId, Math.min(...scores)]))
    const taskIdToMaximumScore = Object.fromEntries(taskIdsWithScores.map(([taskId, scores]) => [taskId, Math.max(...scores)]))

    const averages = ids.map(id => scores.get(id).average)
    const minAverage = Math.min(...averages)
    const maxAverage = Math.max(...averages)

    function getRelativeScore(id, taskId) {
        return (scores.get(id).tasks[taskId] - taskIdToMinimumScore[taskId]) / (taskIdToMaximumScore[taskId] - taskIdToMinimumScore[taskId])
    }

    for (const taskId of taskIds)
        tableHeadE.insertCell().appendChild(createTextE(taskId))

    const evaluationsSortedByAverageScore = Array.from(scores.entries())
        .toSorted(([id1, scores1], [id2, scores2]) => scores2.average - scores1.average)

    for (const [id, { tasks, average }] of evaluationsSortedByAverageScore) {
        const rowE = tableBodyE.insertRow()
        rowE.insertCell().appendChild(createModelLinkE(evaluations.get(id)))

        createTableScoreCell(rowE, createTextE(round(average)), (average - minAverage) / (maxAverage - minAverage))

        rowE.insertCell()

        for (const taskId of taskIds)
            createTableScoreCell(rowE, createTextE(round(tasks[taskId])), getRelativeScore(id, taskId))
    }

    return containerE
}
