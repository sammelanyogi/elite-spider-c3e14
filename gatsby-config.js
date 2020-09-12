const siteMetadata = require('./site-metadata.json')

module.exports = {
	pathPrefix: '/',
	siteMetadata: siteMetadata,
	plugins: [
		`gatsby-plugin-react-helmet`,
		`gatsby-source-data`,
		{
			resolve: `gatsby-transformer-remark`,
			options: {
				plugins: [
					`gatsby-remark-prismjs`,
					{
						resolve: `gatsby-remark-katex`,
						options: {
							// Add any KaTeX options from https://github.com/KaTeX/KaTeX/blob/master/docs/options.md here
							strict: `ignore`
						}
					},
				],
			},
		},
		{
			resolve: `gatsby-source-filesystem`,
			options: {
				name: `pages`,
				path: `${__dirname}/src/pages`
			}
		},
		{
			resolve: `gatsby-plugin-sass`,
			options: {}
		},
		{
			resolve: `gatsby-remark-page-creator`,
			options: {}
		},
		{
			resolve: `@stackbit/gatsby-plugin-menus`,
			options: {
				sourceUrlPath: `fields.url`,
				pageContextProperty: `menus`,
			}
		}
	]
};
