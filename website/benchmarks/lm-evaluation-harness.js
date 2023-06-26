import { createTextE } from '../components/text.js'
import { createBackToMainPageE } from '../components/back-to-main-page.js'
import { createModelsMap, round } from '../utils.js'
import { createModelLinkE } from '../components/model-link.js'

export function computeAverageScore(results) {
    return Object.values(results).map(r => r.acc_norm ?? r.acc).reduce((a, b) => a + b, 0) / Object.values(results).length * 100
}

export async function createV(baseUrl) {
    const containerE = document.createElement('div')

    containerE.appendChild(createBackToMainPageE())

    const otherResultsE = document.createElement('div')
    otherResultsE.classList.add('lm-evaluation-harness__other-results')
    containerE.appendChild(otherResultsE)

    const otherResultsLinkE = document.createElement('a')
    otherResultsLinkE.textContent = 'here'
    otherResultsLinkE.href = 'https://gpt4all.io/index.html'

    otherResultsE.append(
        createTextE('This table is computed using the same method as the one used for the gpt4all leaderboard and therefore the numbers are comparable. '
            + 'You can view the gpt4all leaderboard '),
        otherResultsLinkE,
        createTextE('. Scroll down to the section "Performance Benchmarks".'),
    )

    const models = (await (await fetch(baseUrl + '/__index__.json')).json())
        .filter(model => model.benchmarks.includes('lm-evaluation-harness'))
    const modelsMap = createModelsMap(models)
    const modelNames = models.map(model => model.model_name)

    const results = await Promise.all(modelNames.map(async model =>
        [model, await fetch(baseUrl + '/lm-evaluation-harness/' + model.replace('/', '--') + '.json').then(r => r.json())]))
    const tasks = ['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc_easy', 'arc_challenge', 'openbookqa']
    const resultsMap = Object.fromEntries(results)

    const reportsIndexE = document.createElement('table')
    containerE.appendChild(reportsIndexE)
    const tableHeadE = reportsIndexE.createTHead().insertRow()
    const tableBodyE = reportsIndexE.createTBody()
    tableHeadE.insertCell().appendChild(createTextE('Model'))

    for (const task of tasks)
        tableHeadE.insertCell().appendChild(createTextE(task))
    tableHeadE.insertCell().appendChild(createTextE('Average'))

    const averageScores = Object.fromEntries(modelNames.map(modelName => [modelName, computeAverageScore(resultsMap[modelName].results)]))
    const modelNamesSortedByAverageScore = modelNames.sort((model1Name, model2Name) => averageScores[model2Name] - averageScores[model1Name])

    for (const modelName of modelNamesSortedByAverageScore) {
        const rowE = tableBodyE.insertRow()
        rowE.insertCell().appendChild(createModelLinkE(modelsMap[modelName]))

        for (const task of tasks) {
            let r = resultsMap[modelName].results[task]
            r = (r.acc_norm ?? r.acc) * 100
            rowE.insertCell().appendChild(createTextE(round(r)))
        }

        rowE.insertCell().appendChild(createTextE(round(averageScores[modelName])))
    }

    return containerE
}
