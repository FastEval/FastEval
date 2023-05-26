import { createTextE } from '../components/text.js'
import { round } from '../utils.js'

export function computeAverageScore(results) {
    return Object.values(results).map(r => r.acc_norm ?? r.acc).reduce((a, b) => a + b, 0) / Object.values(results).length * 100
}

export async function createV(baseUrl) {
    const containerE = document.createElement('div')

    const modelNames = await (await fetch(baseUrl + '/__index__.json')).json()

    const results = await Promise.all(modelNames.filter(model => model !== 'gpt-3.5-turbo')
        .map(async model => [model, await fetch(baseUrl + '/lm-evaluation-harness/' + model.replace('/', '--') + '.json').then(r => r.json())]))
    const tasks = Object.keys(results[0][1].results)
    const resultsMap = Object.fromEntries(results)

    const reportsIndexE = document.createElement('table')
    containerE.appendChild(reportsIndexE)
    const tableHeadE = reportsIndexE.createTHead().insertRow()
    const tableBodyE = reportsIndexE.createTBody()
    tableHeadE.insertCell().appendChild(createTextE('Model'))

    for (const task of tasks)
        tableHeadE.insertCell().appendChild(createTextE(task))
    tableHeadE.insertCell().appendChild(createTextE('Average'))

    for (const modelName of modelNames) {
        if (modelName === 'gpt-3.5-turbo')
            continue

        const rowE = tableBodyE.insertRow()
        rowE.insertCell().appendChild(createTextE(modelName))

        for (const task of tasks) {
            let r = resultsMap[modelName].results[task]
            r = (r.acc_norm ?? r.acc) * 100
            rowE.insertCell().appendChild(createTextE(round(r)))
        }

        rowE.insertCell().appendChild(createTextE(round(computeAverageScore(resultsMap[modelName].results))))
    }

    const otherResultsLinkE = document.createElement('a')
    otherResultsLinkE.textContent = 'See also here for numbers for other models. (Section "Performance Benchmarks")'
    otherResultsLinkE.href = 'https://gpt4all.io/index.html'
    containerE.appendChild(otherResultsLinkE)

    return containerE
}
