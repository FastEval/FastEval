import { fetchModels, fetchFiles, allowCharacterLineBreaks, round } from '../utils.js'
import { createTextE } from '../components/text.js'
import { createLinkE } from '../components/link.js'
import { createBackToMainPageE } from '../components/back-to-main-page.js'
import { createConversationItemE } from '../components/conversation-item.js'

export async function createBigBenchHardE(baseUrl, models) {
    const containerE = document.createElement('div')

    containerE.appendChild(createBackToMainPageE('← Back to CoT table', '#?benchmark=cot'))

    const scores = await fetchFiles(baseUrl, models, 'cot', '/scores.json')
    const tasks = Object.keys(scores[0][1].bbh.tasks)

    const tableE = document.createElement('table')
    containerE.appendChild(tableE)
    const tableHeadE = tableE.createTHead().insertRow()
    const tableBodyE = tableE.createTBody()
    tableHeadE.insertCell().appendChild(createTextE('Model'))

    for (const task of tasks)
        tableHeadE.insertCell().appendChild(createTextE(allowCharacterLineBreaks(task)))

    for (const [modelName, modelScores] of scores) {
        const rowE = tableBodyE.insertRow()
        rowE.insertCell().appendChild(createTextE(modelName))
        for (const task of tasks)
            rowE.insertCell().appendChild(createLinkE(round(modelScores.bbh.tasks[task]), { task: 'bbh/' + task, model: modelName }))
    }

    return containerE
}

export async function createTaskV(baseUrl, models, task, parameters) {
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
        createTextE('Model: ' + modelName),
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

    if (parameters.has('task')) {
        containerE.appendChild(await createTaskV(baseUrl, models, parameters.get('task'), parameters))
        return containerE
    }

    containerE.appendChild(createBackToMainPageE())

    const scores = await fetchFiles(baseUrl, models, 'cot', '/scores.json')

    const tableE = document.createElement('table')
    containerE.appendChild(tableE)

    const tableHeadE = tableE.createTHead().insertRow()
    const tableBodyE = tableE.createTBody()
    tableHeadE.insertCell().appendChild(createTextE('Model'))

    const columns = ['gsm8k', 'bbh', 'average']

    for (const column of columns) {
        if (column === 'bbh')
            tableHeadE.insertCell().appendChild(createLinkE('bbh', { task: 'bbh' }))
        else
            tableHeadE.insertCell().appendChild(createTextE(column))
    }

    const sortedScores = scores.sort(([model1Name, model1Scores], [model2Name, model2Scores]) =>
        model2Scores.average - model1Scores.average)

    for (const [modelName, results] of sortedScores) {
        const rowE = tableBodyE.insertRow()
        rowE.insertCell().appendChild(createTextE(modelName))

        for (const column of columns) {
            if (['gsm8k'].includes(column)) {
                rowE.insertCell().appendChild(createLinkE(round(results[column]), { task: column, model: modelName }))
            } else if (column === 'bbh') {
                rowE.insertCell().appendChild(createTextE(round(results[column].average)))
            } else if (column === 'average') {
                rowE.insertCell().appendChild(createTextE(round(results[column])))
            }
        }
    }

    return containerE
}
