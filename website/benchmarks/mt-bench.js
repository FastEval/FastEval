import { fetchModels, fetchFiles, allowCharacterLineBreaks, round, createModelsMap } from '../utils.js'
import { createTextE } from '../components/text.js'
import { createModelLinkE } from '../components/model-link.js'
import { createTableScoreCell } from '../components/table-score-cell.js'
import { createBackToMainPageE } from '../components/back-to-main-page.js'

function computeRelativeScores(scores, categories) {
    const modelNames = Object.keys(scores)

    const relativeScores = {}
    for (const modelName of modelNames)
        relativeScores[modelName] = { categories: {} }

    for (const key of ['average', 'first_turn', 'second_turn']) {
        const values = Object.values(scores).map(scoreValue => scoreValue[key])
        const min = Math.min(...values)
        const max = Math.max(...values)

        for (const modelName of modelNames)
            relativeScores[modelName][key] = (scores[modelName][key] - min) / (max - min)
    }

    for (const category of categories) {
        const values = Object.values(scores).map(scoreValue => scoreValue.categories[category])
        const min = Math.min(...values)
        const max = Math.max(...values)

        for (const modelName of modelNames)
            relativeScores[modelName].categories[category] = (scores[modelName].categories[category] - min) / (max - min)
    }

    return relativeScores
}

export async function createV(baseUrl) {
    const containerE = document.createElement('div')

    containerE.appendChild(createBackToMainPageE())

    const models = await fetchModels(baseUrl)
    const modelsMap = createModelsMap(models)
    const scores = Object.fromEntries(await fetchFiles(baseUrl, models, 'mt-bench', '/scores.json'))
    const sortedScores = Object.fromEntries(Object.entries(scores).toSorted(([model1Name, model1Scores], [model2Name, model2Scores]) =>
        model2Scores.average - model1Scores.average))
    const categories = Object.keys(Object.values(scores)[0].categories)
    const relativeScores = computeRelativeScores(sortedScores, categories)

    const tableE = document.createElement('table')
    containerE.appendChild(tableE)
    const tableHeadE = tableE.createTHead().insertRow()
    const tableBodyE = tableE.createTBody()
    tableHeadE.insertCell().appendChild(createTextE('Model'))
    tableHeadE.insertCell().appendChild(createTextE('Total'))
    tableHeadE.insertCell()
    tableHeadE.insertCell().appendChild(createTextE('1st turn'))
    tableHeadE.insertCell().appendChild(createTextE('2nd turn'))
    tableHeadE.insertCell()

    for (const category of categories) {
        const categoryE = createTextE(allowCharacterLineBreaks(category))
        categoryE.classList.add('vertical')
        tableHeadE.insertCell().appendChild(categoryE)
    }

    for (const [modelName, modelScores] of Object.entries(sortedScores)) {
        const rowE = tableBodyE.insertRow()
        rowE.insertCell().appendChild(createModelLinkE(modelsMap[modelName]))
        createTableScoreCell(rowE, createTextE(round(modelScores.average)), relativeScores[modelName].average)
        rowE.insertCell()
        createTableScoreCell(rowE, createTextE(round(modelScores.first_turn)), relativeScores[modelName].first_turn)
        createTableScoreCell(rowE, createTextE(round(modelScores.second_turn)), relativeScores[modelName].second_turn)
        rowE.insertCell()
        for (const category of categories)
            createTableScoreCell(rowE, createTextE(round(modelScores.categories[category])), relativeScores[modelName].categories[category])
    }

    return containerE
}
