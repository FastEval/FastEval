import { fetchEvaluations, fetchFiles, round } from '../utils.js'
import { createTextE } from '../components/text.js'
import { createModelLinkE } from '../components/model-link.js'
import { createTableScoreCell } from '../components/table-score-cell.js'
import { createLinkE } from '../components/link.js'
import { createBackToMainPageE } from '../components/back-to-main-page.js'

export async function createE(baseUrl) {
    const containerE = document.createElement('div')

    containerE.appendChild(createBackToMainPageE())

    const explanationE = document.createElement('div')
    containerE.appendChild(explanationE)
    explanationE.classList.add('code__explanation')
    explanationE.append(
        createTextE('Both HumanEval+ as well as DS-1000-Chat measure the python coding abilities of a language model. '
            + 'However, the approach in both benchmarks is slightly different. '
            + 'HumanEval+ focuses on writing complete and simple python functions. '
            + 'DS-1000 is more about completing a missing part in code that uses datascience libraries. '
            + 'Click on the corresponding columns for more information.')
    )

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
    const taskIdsWithScores = tasks.map(([taskId, _]) => [taskId, ids.map(id => scores.get(id)[taskId])])
    const taskIdToMinimumScore = Object.fromEntries(taskIdsWithScores.map(([taskId, scores]) => [taskId, Math.min(...scores)]))
    const taskIdToMaximumScore = Object.fromEntries(taskIdsWithScores.map(([taskId, scores]) => [taskId, Math.max(...scores)]))

    const totals = ids.map(id => scores.get(id).code)
    const minTotal = Math.min(...totals)
    const maxTotal = Math.max(...totals)

    function getRelativeScore(id, taskId) {
        return (scores.get(id)[taskId] - taskIdToMinimumScore[taskId]) / (taskIdToMaximumScore[taskId] - taskIdToMinimumScore[taskId])
    }

    for (const [taskId, taskName] of tasks)
        tableHeadE.insertCell().appendChild(createLinkE(taskName, { benchmark: taskId }))

    const evaluationsSortedByAverageScore = Array.from(scores.entries())
        .toSorted(([id1, scores1], [id2, scores2]) => scores2.code - scores1.code)

    for (const [id, scores] of evaluationsSortedByAverageScore) {
        const rowE = tableBodyE.insertRow()
        rowE.insertCell().appendChild(createModelLinkE(evaluations.get(id)))

        createTableScoreCell(rowE, createTextE(round(scores.code)), (scores.code - minTotal) / (maxTotal - minTotal))

        rowE.insertCell()

        for (const [taskId, taskName] of tasks)
            createTableScoreCell(rowE, createTextE(round(scores[taskId])), getRelativeScore(id, taskId))
    }

    console.log(scores)

    return containerE
}
