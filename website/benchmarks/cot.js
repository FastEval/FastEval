import { fetchEvaluations, fetchFiles, allowCharacterLineBreaks, round } from '../utils.js'
import { createTextE } from '../components/text.js'
import { createLinkE } from '../components/link.js'
import { createBackToMainPageE } from '../components/back-to-main-page.js'
import { createConversationItemE } from '../components/conversation-item.js'
import { createModelLinkE } from '../components/model-link.js'
import { createTableScoreCell } from '../components/table-score-cell.js'

export async function createMultiTaskE(baseUrl, evaluations, benchmark) {
    const containerE = document.createElement('div')

    containerE.appendChild(createBackToMainPageE('← Back to CoT table', { 'benchmark': 'cot' }))

    const absoluteScores = await fetchFiles(baseUrl, evaluations, 'cot', '/scores.json')
    const relativeScores = computeRelativeScores(absoluteScores)
    const tasks = Object.keys(Object.entries(relativeScores)[0][1][benchmark].tasks)

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

    const sortedAbsoluteScores = Array.from(absoluteScores.entries())
        .sort(([id1, evaluationScores1], [id2, evaluationScores2]) => evaluationScores2.total - evaluationScores1.total)

    for (const [id, evaluationAbsoluteScores] of sortedAbsoluteScores) {
        const rowE = tableBodyE.insertRow()
        rowE.insertCell().appendChild(createModelLinkE(evaluations.get(id), false))

        createTableScoreCell(
            rowE,
            createTextE(round(evaluationAbsoluteScores[benchmark].average)),
            relativeScores[id][benchmark].average,
        )

        rowE.insertCell()

        for (const task of tasks)
            createTableScoreCell(
                rowE,
                createLinkE(round(evaluationAbsoluteScores[benchmark].tasks[task]), { task: benchmark + '/' + task, id }),
                relativeScores[id][benchmark].tasks[task],
            )
    }

    return containerE
}

export async function createTaskE(baseUrl, evaluations, task, parameters) {
    if (['bbh', 'mmlu'].includes(task))
        return await createMultiTaskE(baseUrl, evaluations, task)

    const id = parameters.get('id')
    const modelName = evaluations.get(id).model_name

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
        createTextE('Model: ', createModelLinkE(evaluations.get(id))),
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

export async function createE(baseUrl, parameters) {
    const containerE = document.createElement('div')

    const evaluations = await fetchEvaluations(baseUrl)

    if (parameters.has('task')) {
        containerE.appendChild(await createTaskE(baseUrl, evaluations, parameters.get('task'), parameters))
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

    const absoluteScores = await fetchFiles(baseUrl, evaluations, 'cot', '/scores.json')
    const relativeScores = computeRelativeScores(absoluteScores)

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

    const sortedAbsoluteScores = Array.from(absoluteScores.entries())
        .sort(([id1, evaluationScores1], [id2, evaluationScores2]) => evaluationScores2.total - evaluationScores1.total)

    for (const [id, evaluationAbsoluteScores] of sortedAbsoluteScores) {
        const rowE = tableBodyE.insertRow()
        rowE.insertCell().appendChild(createModelLinkE(evaluations.get(id)))
        createTableScoreCell(rowE, createTextE(round(evaluationAbsoluteScores.total)), relativeScores[id].total)
        rowE.insertCell()

        for (const [columnId, columnName] of columns) {
            if (columnId === 'gsm8k') {
                const cellE = createLinkE(round(evaluationAbsoluteScores[columnId]), { task: columnId, id })
                createTableScoreCell(rowE, cellE, relativeScores[id][columnId])
            } else if (['bbh', 'mmlu'].includes(columnId)) {
                const cellE = createTextE(round(evaluationAbsoluteScores[columnId].average))
                createTableScoreCell(rowE, cellE, relativeScores[id][columnId].average)
            }
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

        const values = Array.from(absoluteScores.values()).map(e => e[benchmark].average)
        const min = Math.min(...values)
        const max = Math.max(...values)
        for (const id of ids)
            relativeScores[id][benchmark].average = (absoluteScores.get(id)[benchmark].average - min) / (max - min)

        for (const taskName of tasks) {
            const values = Array.from(absoluteScores.values()).map(e => e[benchmark].tasks[taskName])
            const min = Math.min(...values)
            const max = Math.max(...values)
            for (const id of ids)
                relativeScores[id][benchmark].tasks[taskName] = (absoluteScores.get(id)[benchmark].tasks[taskName] - min) / (max - min)
        }
    }

    const total = Array.from(absoluteScores.values()).map(e => e.total)
    const minTotal = Math.min(...total)
    const maxTotal = Math.max(...total)
    for (const id of ids)
        relativeScores[id].total = (absoluteScores.get(id).total - minTotal) / (maxTotal - minTotal)

    return relativeScores
}
