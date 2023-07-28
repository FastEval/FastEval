import { fetchEvaluations, fetchFiles, allowCharacterLineBreaks, round, createEvaluationsMap } from '../utils.js'
import { createTextE } from '../components/text.js'
import { createLinkE } from '../components/link.js'
import { createBackToMainPageE } from '../components/back-to-main-page.js'
import { createConversationItemE } from '../components/conversation-item.js'
import { createModelLinkE } from '../components/model-link.js'
import { createTableScoreCell } from '../components/table-score-cell.js'

export async function createMultiTaskE(baseUrl, evaluations, evaluationsMap, benchmark) {
    const containerE = document.createElement('div')

    containerE.appendChild(createBackToMainPageE('← Back to CoT table', { 'benchmark': 'cot' }))

    const absoluteScores = await fetchFiles(baseUrl, evaluations, 'cot', '/scores.json')
    const relativeScores = Object.entries(computeRelativeScores(absoluteScores))
    const tasks = Object.keys(relativeScores[0][1][benchmark].tasks)

    const tableE = document.createElement('table')
    containerE.appendChild(tableE)
    const tableHeadE = tableE.createTHead().insertRow()
    const tableBodyE = tableE.createTBody()
    tableHeadE.insertCell().appendChild(createTextE('Model'))
    tableHeadE.insertCell().appendChild(createTextE('Total'))
    tableHeadE.insertCell()

    for (const task of tasks) {
        const taskE = createTextE(allowCharacterLineBreaks(task))
        taskE.classList.add('vertical')
        tableHeadE.insertCell().appendChild(taskE)
    }

    const sortedRelativeScores = relativeScores.sort(([id1, evaluationScores1], [id2, evaluationScores2]) =>
        evaluationScores2[benchmark].total - evaluationScores1[benchmark].total)

    for (const [id, evaluationRelativeScores] of sortedRelativeScores) {
        const rowE = tableBodyE.insertRow()
        rowE.insertCell().appendChild(createModelLinkE(evaluationsMap.get(id), false))
        rowE.insertCell().appendChild(createTextE(round(evaluationRelativeScores[benchmark].total)))
        rowE.insertCell()
        for (const task of tasks)
            createTableScoreCell(
                rowE,
                createLinkE(round(absoluteScores.get(id)[benchmark].tasks[task]), { task: benchmark + '/' + task, id }),
                evaluationRelativeScores[benchmark].tasks[task]
            )
    }

    return containerE
}

export async function createTaskV(baseUrl, evaluations, evaluationsMap, task, parameters) {
    if (['bbh', 'mmlu'].includes(task))
        return await createMultiTaskE(baseUrl, evaluations, evaluationsMap, task)

    const id = parameters.get('id')
    const modelName = evaluationsMap.get(id).model_name

    const containerE = document.createElement('div')

    let previousUrl = { 'benchmark': 'cot' }
    if (task.includes('/'))
        previousUrl['task'] = task.split('/').slice(0, -1).join('/')
    containerE.appendChild(createBackToMainPageE('← Back to table', previousUrl))

    const data = await (await fetch(baseUrl + '/cot/' + modelName.replace('/', '--') + '/' + id + '/' + '/tasks/' + task + '.json')).json()

    const infoE = document.createElement('div')
    infoE.classList.add('cot__information')
    containerE.appendChild(infoE)
    infoE.append(
        createTextE('Task: ' + task),
        createTextE('Model: ', createModelLinkE(evaluationsMap.get(id))),
        createTextE('Score: ' + round(data.score)),
    )

    const samplesE = document.createElement('div')
    containerE.appendChild(samplesE)
    samplesE.classList.add('samples')

    const sampleId = parameters.get('sample')

    for (const item of data.model_outputs) {
        if (sampleId !== undefined && parseInt(sampleId) !== item.id)
            continue

        const itemE = document.createElement('div')
        itemE.classList.add('sample')
        samplesE.appendChild(itemE)

        itemE.append(
            createLinkE('ID: ' + item.id, { sample: item.id }),
            createTextE('The following question was asked:'),
            createConversationItemE('user', item.question),
            createTextE('The following answer was expected:'),
            createConversationItemE('assistant', typeof item.correct_answer === 'number'
                ? ('(' + ['A', 'B', 'C', 'D'][item.correct_answer] + ')')
                : item.correct_answer),
            createTextE('The model responded in the following way:'),
            createConversationItemE('assistant', item.model_answer),
            createTextE('This answer was ' + (item.correct ? 'correct' : 'incorrect') + '.'),
        )
    }

    return containerE
}

export async function createV(baseUrl, parameters) {
    const containerE = document.createElement('div')

    const evaluations = await fetchEvaluations(baseUrl)
    const evaluationsMap = createEvaluationsMap(evaluations)

    if (parameters.has('task')) {
        containerE.appendChild(await createTaskV(baseUrl, evaluations, evaluationsMap, parameters.get('task'), parameters))
        return containerE
    }

    containerE.appendChild(createBackToMainPageE())

    const cotHubLinkE = document.createElement('a')
    cotHubLinkE.textContent = 'here'
    cotHubLinkE.href = 'https://github.com/FranxYao/chain-of-thought-hub'
    const explanationE = createTextE('This benchmark measures the CoT (chain-of-thought) reasoning capabilities. '
        + 'It uses a set of questions (depending on the task) and prompts the model to first explain its reasoning step-by-step and then output the answer. '
        + 'The reasoning itself is currently ignored and only the final answer is checked for correctness. '
        + 'For another leaderboard that focuses more on this, see ', cotHubLinkE, '.')
    explanationE.classList.add('cot-explanation')
    containerE.appendChild(explanationE)

    const scores = computeRelativeScores(await fetchFiles(baseUrl, evaluations, 'cot', '/scores.json'))

    const tableE = document.createElement('table')
    containerE.appendChild(tableE)

    const tableHeadE = tableE.createTHead().insertRow()
    const tableBodyE = tableE.createTBody()
    tableHeadE.insertCell().appendChild(createTextE('Model'))

    const columns = [
        ['gsm8k', 'GSM8K'],
        ['bbh', 'BBH'],
        ['mmlu', 'MMLU'],
    ]

    tableHeadE.insertCell().appendChild(createTextE('Total'))
    tableHeadE.insertCell()

    for (const [columnId, columnName] of columns) {
        if (['bbh', 'mmlu'].includes(columnId))
            tableHeadE.insertCell().appendChild(createLinkE(columnName, { task: columnId }))
        else
            tableHeadE.insertCell().appendChild(createTextE(columnName))
    }

    const sortedScores = Object.entries(scores).sort(([id1, evaluationScores1], [id2, evaluationScores2]) =>
        evaluationScores2.total - evaluationScores1.total)

    for (const [id, results] of sortedScores) {
        const rowE = tableBodyE.insertRow()
        rowE.insertCell().appendChild(createModelLinkE(evaluationsMap.get(id)))

        rowE.insertCell().appendChild(createTextE(round(results['total'])))
        rowE.insertCell()

        for (const [columnId, columnName] of columns) {
            const cellE = columnId === 'gsm8k' ? createLinkE(round(results[columnId]), { task: columnId, id })
                : ['bbh', 'mmlu'].includes(columnId) ? createTextE(round(results[columnId].total)) : undefined
            createTableScoreCell(rowE, cellE, results[columnId].total ?? results[columnId])
        }
    }

    return containerE
}

export function computeRelativeScores(absoluteScores) {
    const ids = Array.from(absoluteScores.keys())

    const relativeScores = {}
    for (const id of ids)
        relativeScores[id] = { bbh: { tasks: {} }, mmlu: { tasks: {} } }

    const gsm8k = Array.from(absoluteScores.values()).map(e => e.gsm8k)
    const minGsm8k = Math.min(...gsm8k)
    const maxGsm8k = Math.max(...gsm8k)

    for (const id of ids)
        relativeScores[id].gsm8k = (absoluteScores.get(id).gsm8k - minGsm8k) / (maxGsm8k - minGsm8k)

    for (const benchmark of ['bbh', 'mmlu']) {
        const tasks = Object.keys(absoluteScores.get(ids[0])[benchmark].tasks)

        for (const taskName of tasks) {
            const values = Array.from(absoluteScores.values()).map(e => e[benchmark].tasks[taskName])
            const min = Math.min(...values)
            const max = Math.max(...values)
            for (const id of ids)
                relativeScores[id][benchmark].tasks[taskName] = (absoluteScores.get(id)[benchmark].tasks[taskName] - min) / (max - min)
        }

        for (const id of ids)
            relativeScores[id][benchmark].total = Object.values(relativeScores[id][benchmark].tasks).reduce((a, b) => a + b) / tasks.length

        const scores = Object.values(relativeScores).map(e => e[benchmark].total)
        const minScore = Math.min(...scores)
        const maxScore = Math.max(...scores)

        for (const id of ids)
            relativeScores[id][benchmark].total = (relativeScores[id][benchmark].total - minScore) / (maxScore - minScore)
    }

    for (const id of ids)
        relativeScores[id].total = (relativeScores[id].gsm8k + relativeScores[id].bbh.total + relativeScores[id].mmlu.total) / 3

    return relativeScores
}
