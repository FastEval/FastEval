import { computeUpdatedHash, createModelsMap } from '../utils.js'
import { createConversationItemE } from '../components/conversation-item.js'
import { createModelSelectV } from '../components/model-select.js'
import { createSelectV } from '../components/select.js'
import { createTextE } from '../components/text.js'
import { createBackToMainPageE } from '../components/back-to-main-page.js'
import { createModelLinkE } from '../components/model-link.js'

export async function createV(baseUrl, parameters) {
    const containerE = document.createElement('div')

    containerE.appendChild(createBackToMainPageE())

    const explanationE = document.createElement('div')
    explanationE.classList.add('vicuna__explanation')
    const vicunaLinkE = document.createElement('a')
    vicunaLinkE.textContent = 'here'
    vicunaLinkE.href = 'https://lmsys.org/blog/2023-03-30-vicuna/'
    explanationE.append(
        createTextE('This benchmark queries all models on a set of prompts. It then uses another more capable model (here: GPT-3.5, but more often GPT-4)'
            + ' to review the model outputs, comparing two different models at a time. '
            + 'If you are interested in seeing the results for specific models, you can filter the reviews below. See '),
        vicunaLinkE,
        createTextE(' for more information on this benchmark, though it has been slightly modified.')
    )
    containerE.appendChild(explanationE)

    const models = (await (await fetch(baseUrl + '/__index__.json')).json())
        .filter(model => model.benchmarks.includes('vicuna'))
    const modelsMap = createModelsMap(models)
    const modelNames = models.map(model => model.model_name)

    const filterE = document.createElement('div')
    filterE.classList.add('vicuna__filter')
    containerE.appendChild(filterE)
    const { view: select1V, element: select1E } = createModelSelectV('Model 1', modelsMap, true)
    filterE.appendChild(select1V)
    const { view: select2V, element: select2E } = createModelSelectV('Model 2', modelsMap, true)
    filterE.appendChild(select2V)
    const { view: selectWinnerV, element: selectWinnerE } = createSelectV('Winner Model', ['any', 'Model 1', 'Model 2'], ['any', 'model1', 'model2'])
    filterE.appendChild(selectWinnerV)

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

    const [questions, reviews, ...answers] = await Promise.all([
        fetch('./data/vicuna/questions.json').then(r => r.json()), // TODO Make relative to base url (probably change base url to root)
        fetch(baseUrl + '/vicuna/reviews.json').then(r => r.json()),
        ...modelAnswersToFetch.map(modelName => fetch(baseUrl + '/vicuna/answers/' + modelName.replace('/', '--') + '.json').then(r => r.json())),
    ])

    const modelNameToAnswers = Object.fromEntries(modelAnswersToFetch.map((modelName, index) => [modelName, answers[index]]))

    const samplesE = document.createElement('div')
    containerE.appendChild(samplesE)
    samplesE.classList.add('samples')
    let numberOfRenderedReviews = 0
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

        if (numberOfRenderedReviews > 100)
            break
        numberOfRenderedReviews++

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
            createTextE('Assistant #1 (', createModelLinkE(modelsMap[review.model1]), ') answered this way:'),
            createConversationItemE('assistant', answer1),
            createTextE('Assistant #2 (', createModelLinkE(modelsMap[review.model2]), ') answered this way:'),
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
