import { fetchModels, fetchFiles, allowCharacterLineBreaks, round, createModelsMap } from '../utils.js'
import { createTextE } from '../components/text.js'
import { createLinkE } from '../components/link.js'
import { createBackToMainPageE } from '../components/back-to-main-page.js'
import { createConversationItemE } from '../components/conversation-item.js'
import { createModelLinkE } from '../components/model-link.js'
import { createTableScoreCell } from '../components/table-score-cell.js'

export async function createBigBenchHardE(baseUrl, models) {
    const containerE = document.createElement('div')

    containerE.appendChild(createBackToMainPageE('← Back to CoT table', '#?benchmark=cot'))

    const scores = Object.entries(computeRelativeScores(Object.fromEntries(await fetchFiles(baseUrl, models, 'cot', '/scores.json'))))
    const tasks = Object.keys(scores[0][1].bbh.tasks)

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

    const sortedScores = scores.sort(([model1Name, model1Scores], [model2Name, model2Scores]) =>
        model2Scores.bbh.total - model1Scores.bbh.total)

    const modelsMap = createModelsMap(models)

    for (const [modelName, modelScores] of sortedScores) {
        const rowE = tableBodyE.insertRow()
        rowE.insertCell().appendChild(createModelLinkE(modelsMap[modelName]))
        rowE.insertCell().appendChild(createTextE(round(modelScores.bbh.total)))
        rowE.insertCell()
        for (const task of tasks)
            createTableScoreCell(rowE, createLinkE(round(modelScores.bbh.tasks[task]), { task: 'bbh/' + task, model: modelName }),
                modelScores.bbh.tasks[task])
    }

    return containerE
}

export async function createTaskV(baseUrl, models, modelsMap, task, parameters) {
    if (task === 'bbh')
        return await createBigBenchHardE(baseUrl, models)

    const modelName = parameters.get('model')

    const containerE = document.createElement('div')

    let previousUrl = '#?benchmark=cot'
    if (task.includes('/'))
        previousUrl += '&task=' + task.split('/').slice(0, -1).join('/')
    containerE.appendChild(createBackToMainPageE('← Back to table', previousUrl))

    const data = await (await fetch('reports/cot/' + modelName.replace('/', '--') + '/tasks/' + task + '.json')).json()

    const infoE = document.createElement('div')
    infoE.classList.add('cot__information')
    containerE.appendChild(infoE)
    infoE.append(
        createTextE('Task: ' + task),
        createTextE('Model: ', createModelLinkE(modelsMap[modelName])),
        createTextE('Score: ' + round(data.score)),
    )

    const samplesE = document.createElement('div')
    containerE.appendChild(samplesE)
    samplesE.classList.add('samples')

    for (const item of data.model_outputs) {
        const itemE = document.createElement('div')
        itemE.classList.add('sample')
        samplesE.appendChild(itemE)

        itemE.append(
            createTextE('The following question was asked:'),
            createConversationItemE('user', item.question),
            createTextE('The following answer was expected:'),
            createConversationItemE('assistant', item.correct_answer),
            createTextE('The model responded in the following way:'),
            createConversationItemE('assistant', item.model_answer),
            createTextE('This answer was ' + (item.correct ? 'correct' : 'incorrect') + '.'),
        )
    }

    return containerE
}

export async function createV(baseUrl, parameters) {
    const containerE = document.createElement('div')

    const models = await fetchModels(baseUrl)
    const modelsMap = createModelsMap(models)

    if (parameters.has('task')) {
        containerE.appendChild(await createTaskV(baseUrl, models, modelsMap, parameters.get('task'), parameters))
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

    const scores = computeRelativeScores(Object.fromEntries(await fetchFiles(baseUrl, models, 'cot', '/scores.json')))

    const tableE = document.createElement('table')
    containerE.appendChild(tableE)

    const tableHeadE = tableE.createTHead().insertRow()
    const tableBodyE = tableE.createTBody()
    tableHeadE.insertCell().appendChild(createTextE('Model'))

    const columns = [
        ['gsm8k', 'GSM8K'],
        ['bbh', 'BBH'],
    ]

    tableHeadE.insertCell().appendChild(createTextE('Total'))
    tableHeadE.insertCell()

    for (const [columnId, columnName] of columns) {
        if (columnId === 'bbh')
            tableHeadE.insertCell().appendChild(createLinkE(columnName, { task: 'bbh' }))
        else
            tableHeadE.insertCell().appendChild(createTextE(columnName))
    }

    const sortedScores = Object.entries(scores).sort(([model1Name, model1Scores], [model2Name, model2Scores]) =>
        model2Scores.total - model1Scores.total)

    for (const [modelName, results] of sortedScores) {
        const rowE = tableBodyE.insertRow()
        rowE.insertCell().appendChild(createModelLinkE(modelsMap[modelName]))

        rowE.insertCell().appendChild(createTextE(round(results['total'])))
        rowE.insertCell()

        for (const [columnId, columnName] of columns) {
            const cellE = columnId === 'gsm8k' ? createLinkE(round(results[columnId]), { task: columnId, model: modelName })
                : columnId === 'bbh' ? createTextE(round(results[columnId].total)) : undefined
            createTableScoreCell(rowE, cellE, results[columnId].total ?? results[columnId])
        }
    }

    return containerE
}

export function computeRelativeScores(absoluteScores) {
    const modelNames = Object.keys(absoluteScores)

    const relativeScores = {}
    for (const modelName of modelNames)
        relativeScores[modelName] = { bbh: { tasks: {} } }

    const gsm8k = Object.values(absoluteScores).map(e => e.gsm8k)
    const minGsm8k = Math.min(...gsm8k)
    const maxGsm8k = Math.max(...gsm8k)

    for (const modelName of modelNames)
        relativeScores[modelName].gsm8k = (absoluteScores[modelName].gsm8k - minGsm8k) / (maxGsm8k - minGsm8k)

    const bbhTasks = Object.keys(absoluteScores[modelNames[0]].bbh.tasks)

    for (const taskName of bbhTasks) {
        const values = Object.values(absoluteScores).map(e => e.bbh.tasks[taskName])
        const min = Math.min(...values)
        const max = Math.max(...values)
        for (const modelName of modelNames)
            relativeScores[modelName].bbh.tasks[taskName] = (absoluteScores[modelName].bbh.tasks[taskName] - min) / (max - min)
    }

    for (const modelName of modelNames)
        relativeScores[modelName].bbh.total = Object.values(relativeScores[modelName].bbh.tasks).reduce((a, b) => a + b) / bbhTasks.length

    const bbh = Object.values(relativeScores).map(e => e.bbh.total)
    const minBbh = Math.min(...bbh)
    const maxBbh = Math.max(...bbh)

    for (const modelName of modelNames)
        relativeScores[modelName].bbh.total = (relativeScores[modelName].bbh.total - minBbh) / (maxBbh - minBbh)

    for (const modelName of modelNames)
        relativeScores[modelName].total = (relativeScores[modelName].gsm8k + relativeScores[modelName].bbh.total) / 2

    return relativeScores
}
