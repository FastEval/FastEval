import { computeUpdatedHash } from '../utils.js'
import { createConversationItemE } from '../components/conversation-item.js'
import { createModelSelectV } from '../components/model-select.js'
import { createSelectV } from '../components/select.js'
import { createTextE } from '../components/text.js'

export async function createV(baseUrl, parameters) {
    const containerE = document.createElement('div')

    const modelNames = (await (await fetch(baseUrl + '/__index__.json')).json()).map(m => m[1])

    const { view: select1V, element: select1E } = createModelSelectV('Model 1', ['any', ...modelNames])
    containerE.appendChild(select1V)
    const { view: select2V, element: select2E } = createModelSelectV('Model 2', ['any', ...modelNames])
    containerE.appendChild(select2V)
    const { view: selectWinnerV, element: selectWinnerE } = createSelectV('Winner', ['Any', 'Model 1', 'Model 2'], ['any', 'model1', 'model2'])
    containerE.appendChild(selectWinnerV)

    const model1 = parameters.get('model1') ?? 'any'
    const model2 = parameters.get('model2') ?? 'any'
    const winnerModel = parameters.get('winner') ?? 'any'
    const winnerModelName = winnerModel === 'model1' ? model1 : winnerModel === 'model2' ? model2 : 'any'

    select1E.value = model1.replace('/', '--')
    select2E.value = model2.replace('/', '--')
    selectWinnerE.value = winnerModel

    select1E.addEventListener('change', () => { location.hash = computeUpdatedHash({ model1: select1E.value.replace('--', '/') }) })
    select2E.addEventListener('change', () => { location.hash = computeUpdatedHash({ model2: select2E.value.replace('--', '/') }) })
    selectWinnerE.addEventListener('change', () => { location.hash = computeUpdatedHash({ winner: selectWinnerE.value.replace(' ', '').toLowerCase() }) })

    const modelAnswersToFetch = (model1 === 'any' || model2 === 'any') ? modelNames : [model1, model2]

    const [questions, { reviews }, ...answers] = await Promise.all([
        fetch('./questions.json').then(r => r.json()), // TODO Make relative to base url (probably change base url to root)
        fetch(baseUrl + '/vicuna/reviews.json').then(r => r.json()),
        ...modelAnswersToFetch.map(modelName => fetch(baseUrl + '/vicuna/answers/' + modelName.replace('/', '--') + '.json').then(r => r.json())),
    ])

    const modelNameToAnswers = Object.fromEntries(modelAnswersToFetch.map((modelName, index) => [modelName, answers[index]]))

    const samplesE = document.createElement('div')
    containerE.appendChild(samplesE)
    samplesE.classList.add('samples')
    for (const review of reviews) {
        const reviewIsRelevant = (model1 === 'any' && model2 === 'any')
            || (model1 === 'any' && [review.model1, review.model2].includes(model2))
            || (model2 === 'any' && [review.model1, review.model2].includes(model1))
            || (review.model1 == model1 && review.model2 == model2)
            || (review.model1 == model2 && review.model2 == model1)
        if (!reviewIsRelevant)
            continue

        const reviewWinnerModelName = review['model' + review.winner_model]
        if (winnerModel !== 'any' && winnerModelName !== reviewWinnerModelName)
            continue

        const questionId = review.question_id
        const question = questions[questionId]
        const answer1 = modelNameToAnswers[review.model1][questionId]
        const answer2 = modelNameToAnswers[review.model2][questionId]

        const reviewE = document.createElement('div')
        reviewE.classList.add('sample')
        samplesE.appendChild(reviewE)
        reviewE.append(
            createTextE('The following prompt was given:'),
            createConversationItemE('user', question),
            createTextE('Assistant #1 (' + review.model1 + ') answered this way:'),
            createConversationItemE('assistant', answer1),
            createTextE('Assistant #2 (' + review.model2 + ') answered this way:'),
            createConversationItemE('assistant', answer2),
            createTextE('The following review was given:'),
            createConversationItemE('assistant', review.review),
            review.winner_model === 'tie'
                ? createTextE('Therefore, the result is a tie.')
                : createTextE('Therefore, assistant #' + review.winner_model + ' (' + reviewWinnerModelName + ') won.'),
        )
    }

    return containerE
}
