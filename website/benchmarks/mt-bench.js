import { fetchEvaluations, fetchFiles, round } from '../utils.js'
import { createTextE } from '../components/text.js'
import { createLinkE } from '../components/link.js'
import { createModelLinkE } from '../components/model-link.js'
import { createTableScoreCell } from '../components/table-score-cell.js'
import { createBackToMainPageE } from '../components/back-to-main-page.js'
import { createConversationItemE } from '../components/conversation-item.js'

function computeRelativeScores(absoluteScores, categories) {
    const ids = Object.keys(absoluteScores)

    const relativeScores = {}
    for (const id of ids)
        relativeScores[id] = { categories: {} }

    for (const key of ['average', 'first_turn', 'second_turn']) {
        const values = Object.values(absoluteScores).map(scoreValue => scoreValue[key])
        const min = Math.min(...values)
        const max = Math.max(...values)
        for (const id of ids)
            relativeScores[id][key] = (absoluteScores[id][key] - min) / (max - min)
    }

    for (const category of categories) {
        const values = Object.values(absoluteScores).map(scoreValue => scoreValue.categories[category])
        const min = Math.min(...values)
        const max = Math.max(...values)
        for (const id of ids)
            relativeScores[id].categories[category] = (absoluteScores[id].categories[category] - min) / (max - min)
    }

    return relativeScores
}

export async function createEvaluationCategoryE({ baseUrl, id, category, evaluations }) {
    const containerE = document.createElement('div')

    containerE.appendChild(createBackToMainPageE('← Back to MT-Bench table', { 'benchmark': 'mt-bench' }))

    const modelName = evaluations.get(id).model_name
    const folderName = modelName.replace('/', '--')

    const [questions, answers, judgeReplies, scores] = await Promise.all([
        fetch(baseUrl + '/../data/mt-bench/questions.json').then(r => r.json()),
        fetch(baseUrl + '/mt-bench/' + folderName + '/' + id + '/answers.json').then(r => r.json()),
        fetch(baseUrl + '/mt-bench/' + folderName + '/' + id + '/judge-replies.json').then(r => r.json()),
        fetch(baseUrl + '/mt-bench/' + folderName + '/' + id + '/scores.json').then(r => r.json()),
    ])

    const infoE = document.createElement('div')
    infoE.classList.add('mt-bench__information')
    containerE.appendChild(infoE)
    infoE.append(
        createTextE('Model: ' + modelName),
        createTextE('Category: ' + category),
        createTextE('Category score: ' + scores.categories[category] + '/10')
    )

    const questionsWithCategory = Object.entries(questions)
        .filter(([questionId, question]) => question.category == category)

    const samplesE = document.createElement('div')
    samplesE.classList.add('samples')
    containerE.appendChild(samplesE)

    for (const [questionId, question] of questionsWithCategory) {
        const answer = answers[questionId]
        const judgeReply = judgeReplies.filter(item => item.question_id === questionId)
        const judgeReply1 = judgeReply.filter(item => item.turn_number === 0)[0]
        const judgeReply2 = judgeReply.filter(item => item.turn_number === 1)[0]

        const sampleE = document.createElement('div')
        sampleE.classList.add('sample')
        samplesE.appendChild(sampleE)
        sampleE.append(
            createLinkE('ID: ' + questionId, { question: questionId }),
            createTextE('The model was given the following question'),
            createConversationItemE('user', question.turns[0]),
            createTextE('The model gave the following answer:'),
            createConversationItemE('assistant', answer[0]),
            createTextE('GPT-4 gave the following judgement:'),
            createConversationItemE('assistant', judgeReply1.judge_reply),
            createTextE('The model was given the followup question'),
            createConversationItemE('user', question.turns[1]),
            createTextE('The model gave the following answer:'),
            createConversationItemE('assistant', answer[1]),
            createTextE('GPT-4 gave the following judgement:'),
            createConversationItemE('assistant', judgeReply2.judge_reply),
        )
    }

    return containerE
}

export async function createE(baseUrl, parameters) {
    const evaluations = await fetchEvaluations(baseUrl)

    if (parameters.has('id') && parameters.has('category'))
        return await createEvaluationCategoryE({
            baseUrl,
            id: parameters.get('id'),
            category: parameters.get('category'),
            evaluations,
        })

    const containerE = document.createElement('div')

    containerE.appendChild(createBackToMainPageE())

    const mtBenchLinkE = document.createElement('a')
    mtBenchLinkE.textContent = 'MT-Bench'
    mtBenchLinkE.href = 'https://arxiv.org/abs/2306.05685'

    const lmSysLeaderboardLinkE = document.createElement('a')
    lmSysLeaderboardLinkE.textContent = 'LMSys Leaderboard'
    lmSysLeaderboardLinkE.href = 'https://chat.lmsys.org/?leaderboard'

    const explanationE = createTextE(mtBenchLinkE, ' measures conversational capabilities. '
        + 'It evaluates a LLM on a set of 80 conversations with two turns each. '
        + 'The 160 model outputs are then rated by GPT-4. '
        + 'The evaluation method should be the same as for the ', lmSysLeaderboardLinkE, ' so the results should be comparable. '
        + 'Note however that due to differences in the model prompt templates and the low number of samples the numbers may be slightly different. ')
    explanationE.classList.add('mt-bench-explanation')
    containerE.appendChild(explanationE)

    const scores = Object.fromEntries(await fetchFiles(baseUrl, evaluations, 'mt-bench', '/scores.json'))
    const sortedScores = Object.fromEntries(Object.entries(scores).toSorted(([id1, evaluation1Scores], [id2, evaluation2Scores]) =>
        evaluation2Scores.average - evaluation1Scores.average))
    const categories = Object.keys(Object.values(scores)[0].categories)
    const relativeScores = computeRelativeScores(sortedScores, categories)

    const tableE = document.createElement('table')
    containerE.appendChild(tableE)
    const tableHeadE = tableE.createTHead().insertRow()
    const tableBodyE = tableE.createTBody()
    tableHeadE.insertCell().appendChild(createTextE('Model'))
    tableHeadE.insertCell().appendChild(createTextE('Total'))
    tableHeadE.insertCell()
    const firstTurnE = createTextE('1st turn')
    firstTurnE.classList.add('vertical')
    tableHeadE.insertCell().appendChild(firstTurnE)
    const secondTurnE = createTextE('2nd turn')
    secondTurnE.classList.add('vertical')
    tableHeadE.insertCell().appendChild(secondTurnE)
    tableHeadE.insertCell()

    for (const category of categories) {
        const categoryE = createTextE(category)
        categoryE.classList.add('vertical')
        tableHeadE.insertCell().appendChild(categoryE)
    }

    for (const [id, evaluationScores] of Object.entries(sortedScores)) {
        const rowE = tableBodyE.insertRow()
        rowE.insertCell().appendChild(createModelLinkE(evaluations.get(id)))
        createTableScoreCell(rowE, createTextE(round(evaluationScores.average)), relativeScores[id].average)
        rowE.insertCell()
        createTableScoreCell(rowE, createTextE(round(evaluationScores.first_turn)), relativeScores[id].first_turn)
        createTableScoreCell(rowE, createTextE(round(evaluationScores.second_turn)), relativeScores[id].second_turn)
        rowE.insertCell()
        for (const category of categories)
            createTableScoreCell(
                rowE,
                createLinkE(round(evaluationScores.categories[category]), { id, category }),
                relativeScores[id].categories[category]
            )
    }

    return containerE
}
