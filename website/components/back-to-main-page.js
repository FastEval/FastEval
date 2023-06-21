export function createBackToMainPageE(text='← Back to main page', link='#') {
    const linkE = document.createElement('a')
    linkE.classList.add('back-to-main-page')
    linkE.textContent = text
    linkE.href = link
    return linkE
}
