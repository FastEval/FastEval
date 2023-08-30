import { fetchEvaluations, fetchFiles, round } from '../utils.js'
import { createTextE } from '../components/text.js'
import { createModelLinkE } from '../components/model-link.js'
import { createTableScoreCell } from '../components/table-score-cell.js'
import { createBackToMainPageE } from '../components/back-to-main-page.js'

export async function createE(baseUrl) {
    const containerE = document.createElement('div')

    containerE.appendChild(createBackToMainPageE('← Back to code table', { benchmark: 'code' }))

    const explanationE = document.createElement('div')
    containerE.appendChild(explanationE)
    explanationE.classList.add('ds1000__explanation')
    const ds1000LinkE = document.createElement('a')
    ds1000LinkE.textContent = 'DS-1000'
    ds1000LinkE.href = 'https://ds1000-code-gen.github.io/'
    explanationE.append(
        ds1000LinkE,
        createTextE(' is a benchmark for evaluating the abilities of a model to understand python data science code. '
            + 'The benchmark mostly (except matplotlib) consists of tasks where the model has to insert a missing piece into existing incomplete code. '
            + 'The original DS-1000 benchmark is only designed for models that have been trained with special tokens on insertion & completion. '
            + 'Here, we have modified it so that it can work with any chat language model.'),
    )

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
