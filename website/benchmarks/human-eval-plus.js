import { createConversationItemE } from '../components/conversation-item.js'
import { createTextE } from '../components/text.js'
import { createLinkE } from '../components/link.js'
import { createBackToMainPageE } from '../components/back-to-main-page.js'
import { computeUpdatedHash, fetchEvaluations, fetchFiles, createEvaluationsMap, round } from '../utils.js'
import { createModelLinkE } from '../components/model-link.js'
import { createTableScoreCell } from '../components/table-score-cell.js'

export async function createV(baseUrl, parameters) {
    const evaluations = await fetchEvaluations(baseUrl)
    const evaluationsMap = createEvaluationsMap(evaluations)

    if (parameters.has('id'))
        return await createModelV(baseUrl, parameters, evaluations, evaluationsMap)

    const containerE = document.createElement('div')

    containerE.appendChild(createBackToMainPageE())

    const explanationE = document.createElement('div')
    containerE.appendChild(explanationE)
    explanationE.classList.add('human-eval-plus__explanation')
    const evalPlusLinkE = document.createElement('a')
    evalPlusLinkE.textContent = 'HumanEval+'
    evalPlusLinkE.href = 'https://github.com/evalplus/evalplus'
    const humanEvalLinkE = document.createElement('a')
    humanEvalLinkE.textContent = 'HumanEval'
    humanEvalLinkE.href = 'https://github.com/openai/human-eval'
    explanationE.append(
        evalPlusLinkE,
        createTextE(' is a benchmark for evaluating the Python coding performance of language models. It is based on '),
        humanEvalLinkE,
        createTextE(' from OpenAI and improves it with more tests. '
            + 'The benchmark provides the LLM with the beginning of a function including the docstring and asks it to complete the code. '
            + 'The resulting model output is then evaluated by running it against a number of tests.')
    )

    const scores = await fetchFiles(baseUrl, evaluations, 'human-eval-plus', '/scores.json')
    const sortedScores = Array.from(scores.entries())
        .toSorted(([id1, evaluationScores1], [id2, evaluationScores2]) => evaluationScores2.scores.plus - evaluationScores1.scores.plus)
    const scoresMap = Object.fromEntries(scores)

    const tableE = document.createElement('table')
    containerE.appendChild(tableE)

    const tableHeadE = tableE.createTHead().insertRow()
    const tableBodyE = tableE.createTBody()
    tableHeadE.insertCell().appendChild(createTextE('Model'))
    tableHeadE.insertCell().appendChild(createTextE('HumanEval+'))
    tableHeadE.insertCell().appendChild(createTextE('HumanEval'))

    const relativeScores = {}
    for (const { id } of evaluations)
        relativeScores[id] = {}
    for (const column of ['base', 'plus']) {
        const columnScores = sortedScores.map(e => e[1].scores[column])
        const min = Math.min(...columnScores)
        const max = Math.max(...columnScores)
        for (const [id, _] of sortedScores)
            relativeScores[id][column] = (scoresMap[id].scores[column] - min) / (max - min)
    }

    for (const [id, modelScores] of sortedScores) {
        const rowE = tableBodyE.insertRow()
        rowE.insertCell().appendChild(createModelLinkE(evaluationsMap.get(id)))
        createTableScoreCell(rowE, createLinkE(round(modelScores.scores.plus), { id }), relativeScores[id].plus)
        createTableScoreCell(rowE, createTextE(round(modelScores.scores.base)), relativeScores[id].base)
    }

    return containerE
}

export async function createModelV(baseUrl, parameters, evaluations, evaluationsMap) {
    const containerE = document.createElement('div')

    containerE.appendChild(createBackToMainPageE('← Back to HumanEval+ table', { 'benchmark': 'human-eval-plus' }))

    const samplesE = document.createElement('div')
    containerE.appendChild(samplesE)
    samplesE.classList.add('samples')

    const id = parameters.get('id')
    const modelName = evaluationsMap.get(id).model_name
    const folderName = modelName.replace('/', '--')

    const [answers, scores] = await Promise.all([
        fetch(baseUrl + '/human-eval-plus/' + folderName + '/' + id + '/answers.json').then(r => r.json()),
        fetch(baseUrl + '/human-eval-plus/' + folderName + '/' + id + '/scores.json').then(r => r.json()),
    ])

    const merged = { answers: {}, scores: scores.scores }
    for (let i = 0; i < answers.length; i++) {
        const answer = answers[i]
        const taskId = answer.task_id
        const prompt = answer.prompt
        const completionRaw = answer.completion_raw
        const completionProcessed = answer.completion_processed
        const success = scores.answers[i].success
        if (!(taskId in merged.answers))
            merged.answers[taskId] = { prompt, completions: [] }
        merged.answers[taskId].completions.push({ completionRaw, completionProcessed, success })
    }

    for (const [itemId, { prompt, completions: items }] of Object.entries(merged.answers)) {
        if (parameters.has('sample') && parameters.get('sample') !== itemId)
            continue

        const itemE = document.createElement('div')
        itemE.classList.add('sample')
        samplesE.appendChild(itemE)

        function replace(completionNumber) {
            if (parameters.has('sample') && (!parameters.has('completion') || parseInt(parameters.get('completion')) !== (completionNumber + 1)))
                location = '#' + computeUpdatedHash({ completion: Math.min(Math.max(1, completionNumber + 1), items.length) })

            completionNumber = Math.min(Math.max(0, completionNumber), items.length - 1)

            const switchCompletionE = document.createElement('div')
            switchCompletionE.classList.add('switch-completion')

            const currentCompletionE = document.createElement('span')
            switchCompletionE.appendChild(currentCompletionE)
            currentCompletionE.textContent = ((completionNumber + 1) + '/' + items.length)

            const decreaseCompletionNumberE = document.createElement('button')
            decreaseCompletionNumberE.textContent = '<'
            switchCompletionE.prepend(decreaseCompletionNumberE)

            decreaseCompletionNumberE.addEventListener('click', () => { replace(completionNumber - 1) })

            const increaseCompletionNumberE = document.createElement('button')
            increaseCompletionNumberE.textContent = '>'
            switchCompletionE.appendChild(increaseCompletionNumberE)

            increaseCompletionNumberE.addEventListener('click', () => { replace(completionNumber + 1) })

            let successText
            const success = items[completionNumber].success
            if (success.plus && success.base)
                successText = 'This code passed all of the tests.'
            else if (!success.plus && success.base)
                successText = 'This code passed all of the tests in HumanEval, but failed some tests in HumanEval+.'
            else if (success.plus && !success.base)
                successText = 'This code passed all of the tests in HumanEval+, but failed some tests in HumanEval.'
            else if (!success.plus && !success.base)
                successText = 'This code failed some of the tests.'

            itemE.replaceChildren(
                createLinkE('ID: ' + itemId, { sample: itemId }),
                switchCompletionE,
                createTextE('The model was supposed to complete the following code:'),
                createConversationItemE('user', prompt),
                createTextE('The model gave the following code as output:'),
                createConversationItemE('assistant', items[completionNumber].completionRaw.trim()),
                createTextE('The following code was extracted:'),
                createConversationItemE('assistant', items[completionNumber].completionProcessed.trim()),
                createTextE(successText),
            )
        }

        if (parameters.has('completion'))
            replace(parseInt(parameters.get('completion')) - 1)
        else
            replace(0)
    }

    return containerE
}
